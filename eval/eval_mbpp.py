import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import argparse
from gen import GenEvaluation
from datasets import load_dataset
from generic.special_tokens import *
from generic.utils import data_args, openai_args, get_openai_kwargs, decorate_code
from utils import postprocess_output_wf

parser = argparse.ArgumentParser()
parser = data_args(parser)
parser = openai_args(parser)
parser.add_argument("--use_tab", action="store_true", help="Whether to use automated editing")
parser.add_argument("--use_inline", action="store_true", help="Whether to use inline chat")
parser.add_argument("--use_chat", action="store_true", help="Whether to use chat")
args = parser.parse_args()
openai_kwargs = get_openai_kwargs(args)

dataset = load_dataset(args.input_path)
conversations = []
prompts = []
task_ids = []
for sample in dataset["test"]:
    task_ids.append(sample["task_id"])
    prompt = f'"""\n{sample["prompt"]}\n{sample["test_list"][0]}\n"""\n'
    prompts.append(prompt)
    if args.use_tab:
        conversations.append({"conversation": [{"role": "current", "content": decorate_code(prompt, lang="python")}]})
    elif args.use_inline:
        conversations.append({"conversation": [{"role": "current", "content": decorate_code(prompt, lang="python")}, {"role": "user", "content": "Please complete the function."}]})
    elif args.use_chat:
        conversations.append({"conversation": [{"role": "user", "content": f"Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:\n```python\n{prompt}\n```"}]})
    else:
        raise ValueError("Invalid model type: {}".format(args.model_type))

answers = []

with open(args.model_map, 'r') as f:
    model_map = json.load(f)

if args.use_tab or args.use_inline:
    openai_kwargs["extra_body"] = {"skip_special_tokens": False, 'chat_template': 'assistant-conversation'}
openai_kwargs["stop"] = [NEXT_END]

Gen = GenEvaluation(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

results = Gen.gen(conversations)

if args.use_tab:
    output_data = [{"task_id": f"Mbpp/{task_id}", "solution": postprocess_output_wf(current, answer["output"])} for task_id, current, answer in zip(task_ids, prompts, results)]
elif args.use_inline:
    output_data = [{"task_id": f"Mbpp/{task_id}", "solution": postprocess_output_wf(current, answer["output"])} for task_id, current, answer in zip(task_ids, prompts, results)]
elif args.use_instruct:
    output_data = [{"task_id": f"Mbpp/{task_id}", "solution": answer["output"]} for task_id, answer in zip(task_ids, results)]
else:
    raise ValueError("Invalid model type: {}".format(args.model_type))

with open(args.output_path, 'w') as f:
    for item in output_data:
        json_line = json.dumps(item)
        f.write(json_line + '\n')
