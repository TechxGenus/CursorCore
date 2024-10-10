import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import copy
import json
import numba
import torch
import numpy as np
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from transformers import DefaultDataCollator, default_data_collator
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_INDEX = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    num_proc: int = field(default=8, metadata={"help": "Number of processes to use for data preprocessing."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=10000, metadata={"help": "Maximum sequence length."})
    batch_max_length: int = field(default=25000, metadata={"help": "Maximum batch length."})

def preprocess(
    list_data_dict: List,
    tokenizer: transformers.PreTrainedTokenizer,
    num_proc: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [tokenizer.apply_chat_template(l, tokenize=False) for l in list_data_dict]
    sources = [tokenizer.apply_chat_template(l[:-1], tokenize=False, add_generation_prompt=True) for l in list_data_dict]

    """Tokenize a list of strings."""
    def tokenize_text(text):
        return tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        examples_tokenized = list(executor.map(tokenize_text, examples))
        sources_tokenized = list(executor.map(tokenize_text, sources))

    input_ids = [tokenized.input_ids[0].tolist() for tokenized in examples_tokenized]
    labels = copy.deepcopy(input_ids)
    source_lens = [len(tokenized.input_ids[0]) for tokenized in sources_tokenized]
    for label, source_len in zip(labels, source_lens):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(input_ids=input_ids, labels=labels)

@numba.njit
def ffd(lengths, batch_max_length):
    """
    First-Fit Decreasing (FFD) algorithm for bin packing.

    This function sorts the input lengths in decreasing order and then attempts to pack them into bins
    such that the sum of lengths in each bin does not exceed the specified batch_max_length. It returns
    the indices of the original lengths in each bin.

    Args:
        lengths (list or array-like): A list or array of lengths to be packed into bins.
        batch_max_length (int): The maximum allowable length for each bin.

    Returns:
        list of lists: A list where each sublist contains the indices of the original lengths that have
                       been packed into the corresponding bin.
    """
    lengths = np.array(lengths)
    indices = np.argsort(lengths)[::-1]
    lengths = lengths[indices]
    bins = []
    bins_result = []
    for lengths_id, size in enumerate(lengths):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[lengths_id])
                add_new = False
                break
        if add_new:
            bins.append(batch_max_length - size)
            bins_result.append([indices[lengths_id]])
    return bins_result

class PackSampler(Sampler):
    def __init__(self, batch_max_length: int, lengths: List[int], seed: int = 0):
        batches = ffd(lengths, batch_max_length)
        indices = np.random.default_rng(seed=seed).permutation(len(batches))
        self.batches = [batches[idx] for idx in indices]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_proc: int):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as json_file:
            list_data_dict = json.load(json_file)
        data_dict = preprocess(list_data_dict, tokenizer, num_proc=num_proc)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, List[int]]:
        return [dict(input_ids=self.input_ids[i], labels=self.labels[i]) for i in index]

@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, *args, return_position_ids=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids

    def __call__(self, features, return_tensors=None):
        features = [item for feature in features for item in feature]
        if return_tensors is None:
            return_tensors = self.return_tensors
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": []}
        if self.return_position_ids:
            ret.update({"position_ids": []})
        for idx in range(0, len(features)):
            ret["input_ids"] += features[idx]["input_ids"]
            if is_labels_provided:
                ret["labels"] += [IGNORE_INDEX] + features[idx]["labels"][1:]
            else:
                ret["labels"] += [IGNORE_INDEX] + features[idx]["input_ids"][1:]
            if self.return_position_ids:
                ret["position_ids"] += list(range(len(features[idx]["input_ids"])))
        return default_data_collator([ret], return_tensors)

class CustomTrainer(Trainer):
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training [`~torch.utils.data.DataLoader`]."""
        dataloader_params = {
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "sampler": self.sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor
        }
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, num_proc=data_args.num_proc)
    data_collator = DataCollatorWithFlattening()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    lengths = [len(input_id) for input_id in data_module["train_dataset"].input_ids]
    sampler = PackSampler(batch_max_length=training_args.batch_max_length, lengths=lengths, seed=training_args.seed)
    trainer = CustomTrainer(sampler=sampler, model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False
    # model.model.forward = torch.compile(model.model.forward)
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
