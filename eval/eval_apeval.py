import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import argparse
from gen import GenEvaluation
from generic.special_tokens import *
from generic.utils import data_args, openai_args, get_openai_kwargs
from utils import prepare_input_for_wf, prepare_input_for_lc, prepare_input_for_sr, prepare_input_for_base, prepare_input_for_instruct, postprocess_output_wf, postprocess_output_lc, postprocess_output_sr, postprocess_output_instruct, postprocess_output_base

parser = argparse.ArgumentParser()
parser = data_args(parser)
parser = openai_args(parser)
parser.add_argument("--use_target_area", action="store_true", help="Whether to use target area")
parser.add_argument("--sliding_window", type=int, default=-1, help="Sliding window size")
parser.add_argument("--use_wf", action="store_true", help="Whether to use Whole File model")
parser.add_argument("--use_lc", action="store_true", help="Whether to use Locate and Change model")
parser.add_argument("--use_sr", action="store_true", help="Whether to use Search and Replace model")
parser.add_argument("--use_instruct", action="store_true", help="Whether to use instruct model")
parser.add_argument("--use_base", action="store_true", help="Whether to use base model")
args = parser.parse_args()
openai_kwargs = get_openai_kwargs(args)

with open(args.input_path, 'r') as f:
    dataset = json.load(f)

conversations = []
currents = []
task_ids = []
for sample in dataset:
    task_ids.append(sample["task_id"])
    currents.append(sample["current"]["code"])
    if args.use_target_area:
        if "area" in sample:
            if type(sample["area"]) == int:
                sample["current"]["code"] = sample["current"]["code"][:sample["area"]] + TARGET + sample["current"]["code"][sample["area"]:]
            elif type(sample["area"]) == list:
                start, end = sample["area"]
                sample["current"]["code"] = sample["current"]["code"][:end] + TARGET_END + sample["current"]["code"][end:]
                sample["current"]["code"] = sample["current"]["code"][:start] + TARGET_START + sample["current"]["code"][start:]
            else:
                raise ValueError("Invalid area type: {}".format(type(sample["area"])))
    if args.sliding_window != -1:
        assert args.sliding_window > 0, "Sliding window size must be greater than 0"
        if len(sample["history"]) >= args.sliding_window:
            sample["history"] = sample["history"][-args.sliding_window:]
    if args.use_wf:
        conversations.append({"conversation": prepare_input_for_wf(sample)})
    elif args.use_lc:
        conversations.append({"conversation": prepare_input_for_lc(sample)})
    elif args.use_sr:
        conversations.append({"conversation": prepare_input_for_sr(sample)})
    elif args.use_instruct:
        conversations.append({"conversation": prepare_input_for_instruct(sample)})
    elif args.use_base:
        conversations.append({"conversation": prepare_input_for_base(sample)})
    else:
        raise ValueError("Invalid model type: {}".format(args.model_type))

answers = []

with open(args.model_map, 'r') as f:
    model_map = json.load(f)

if args.use_wf or args.use_lc or args.use_sr:
    openai_kwargs["extra_body"] = {"skip_special_tokens": False, 'chat_template': 'assistant-conversation'}
openai_kwargs["stop"] = [NEXT_END]

Gen = GenEvaluation(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

if not args.use_base:
    results = Gen.gen(conversations)
else:
    results = Gen.gen(conversations, api_type="completion")

if args.use_wf:
    output_data = [{"task_id": task_id, "solution": postprocess_output_wf(current, answer["output"])} for task_id, current, answer in zip(task_ids, currents, results)]
elif args.use_lc:
    output_data = [{"task_id": task_id, "solution": postprocess_output_lc(current, answer["output"])} for task_id, current, answer in zip(task_ids, currents, results)]
elif args.use_sr:
    output_data = [{"task_id": task_id, "solution": postprocess_output_sr(current, answer["output"])} for task_id, current, answer in zip(task_ids, currents, results)]
elif args.use_instruct:
    output_data = [{"task_id": task_id, "solution": postprocess_output_instruct(current, answer["output"])} for task_id, current, answer in zip(task_ids, currents, results)]
elif args.use_base:
    output_data = [{"task_id": task_id, "solution": postprocess_output_base(current, answer["output"])} for task_id, current, answer in zip(task_ids, currents, results)]
else:
    raise ValueError("Invalid model type: {}".format(args.model_type))

with open(args.output_path, 'w') as f:
    for item in output_data:
        json_line = json.dumps(item)
        f.write(json_line + '\n')
