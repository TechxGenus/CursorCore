import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import torch
import argparse
import transformers
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from generic.special_tokens import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="The model checkpoint for the model to be merged.")
parser.add_argument("--save_path", type=str, help="The path to save the merged model.")
args = parser.parse_args()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"eos_token": IM_END})
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=dict(additional_special_tokens=SPECIAL_WORDS),
    tokenizer=tokenizer,
    model=model,
)

tokenizer.chat_template = [
    {
        "name": "default",
        "template": "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful programming assistant.<|im_end|>\n' }}{% endif %}{% if not loop.first %}{{ '\n' }}{% endif %}{{ '<|im_start|>' + message['role'] }}{% if 'name' in message %}{{ ' name=' + message['name'] }}{% endif %}{{ '\n' + message['content'] + '<|im_end|>' }}{% if loop.last and add_generation_prompt %}{{ '\n<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
    },
    {
        "name": "assistant-conversation",
        "template": "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful programming assistant.<|im_end|>\n' }}{% endif %}{% if not loop.first %}{{ '\n' }}{% endif %}{{ '<|im_start|>' + message['role'] }}{% if 'name' in message %}{{ ' name=' + message['name'] }}{% endif %}{{ '\n' + message['content'] + '<|im_end|>' }}{% if loop.last and add_generation_prompt %}{{ '\n<|im_start|>assistant\n<|next_start|>' }}{% endif %}{% endfor %}"
    },
    {
        "name": "prefix_response",
        "template": "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful programming assistant.<|im_end|>\n' }}{% endif %}{% if not loop.first %}{{ '\n' }}{% endif %}{{ '<|im_start|>' + message['role'] }}{% if 'name' in message %}{{ ' name=' + message['name'] }}{% endif %}{{ '\n' + message['content'] }}{% if not loop.last %}{{ '<|im_end|>' }}{% endif %}{% endfor %}"
    },
]

model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

tokenizer.save_pretrained(args.save_path)
model.save_pretrained(args.save_path)
