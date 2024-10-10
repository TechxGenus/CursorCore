import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import random
import argparse
import concurrent.futures
from tqdm import tqdm
from gen import GenJudgement, GenInstruction, GenChat
from generic.utils import data_args, openai_args, get_openai_kwargs, decorate_code
from utils import generate_diff, random_edit_series, generate_diff_blocks, apply_selected_blocks

def generate_record_and_change(history, current, final):
    """
    Generates a record of changes between historical code segments, the current code segment, 
    and the final code segment.

    Args:
        history (list): A list of dictionaries representing historical code segments. Each dictionary 
                        should have a 'type' key with the value 'code' and a 'code' key with the code 
                        segment as its value.
        current (dict): A dictionary representing the current code segment. It should have a 'type' key 
                        with the value 'code', a 'code' key with the code segment, and a 'lang' key 
                        indicating the programming language.
        final (str): The final code segment as a string.

    Returns:
        tuple: A tuple containing:
            - blocks (list): A list of blocks representing the differences between the current code 
                             and the final code.
            - current_record (list): A list of dictionaries representing the historical changes and 
                                     the current code segment.
            - future_change (list): A list of decorated code segments representing the future changes 
                                    needed to reach the final code.
    """
    current_record = []
    future_change = []

    history_current = history + [current]
    for previous, now in zip(history_current[:-1], history_current[1:]):
        if previous['type'] == 'code' and now['type'] == 'code':
            diff = generate_diff(previous['code'], now['code'])
            current_record.append({"type": "history", "history": decorate_code(diff, "diff")})
        else:
            raise ValueError("History should only contain code segments now.")

    current_record.append({"type": "current", "current": decorate_code(current['code'], current['lang'], use_line_num=True)})
    blocks = []
    if current['code'] != final:
        blocks = generate_diff_blocks(current['code'], final)
        for i in range(len(blocks)):
            code = apply_selected_blocks(current['code'], blocks, [i])
            future_change.append(decorate_code(generate_diff(current['code'], code), "diff"))
    return blocks, current_record, future_change

parser = argparse.ArgumentParser()
parser = data_args(parser)
parser = openai_args(parser)
parser.add_argument("--alpha", type=float, default=1.0, help="Sampling parameter for history")
parser.add_argument("--data_type", type=str, help="Data Type (aiprogrammer, commit, submit)")
parser.add_argument("--max_per_sample", type=int, default=1, help="Maximum number of samples per input")
parser.add_argument("--use_history_prob", type=float, default=0.6, help="Probability of using history")
parser.add_argument("--use_user_prob", type=float, default=0.5, help="Probability of using user")
parser.add_argument("--limit_one_block_prob", type=float, default=0.5, help="Probability of limiting one block during History")
args = parser.parse_args()
openai_kwargs = get_openai_kwargs(args)

with open(args.model_map, 'r') as f:
    model_map = json.load(f)

with open(args.input_path, 'r') as f:
    inputs = json.load(f)

random.shuffle(inputs)

GenJudge = GenJudgement(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

GenInst = GenInstruction(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

GenCht = GenChat(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

def process(sample):
    """
    Processes a given sample based on the specified data type and generates a series of code transformations.

    Args:
        sample (dict): A dictionary containing the sample data. The structure of the sample depends on the data type:
            - For 'aiprogrammer': Must contain 'output'.
            - For 'commit': Must contain 'code1' and 'code2'.
            - For 'submit': Must contain 'submissions', 'language', and 'problems_contexts'.

    Returns:
        list: A list of dictionaries, each representing a step in the code transformation process. Each dictionary contains:
            - 'history': The history of code transformations up to the current step.
            - 'current': The current code block being processed.
            - 'user': The user input or instruction for the current step.
            - 'chat': The chat or commentary generated for the current step.
            - 'next': The next code block after applying the transformation.
    
    Raises:
        ValueError: If the data type specified in args.data_type is not supported.
    """
    if args.data_type == 'aiprogrammer':
        sample['lang'] = ""
        code_series = []
        for code1, code2 in zip(sample['output'][:-1], sample['output'][1:]):
            if random.random() < args.limit_one_block_prob:
                code_series += random_edit_series(code1, code2)[:-1]
            else:
                code_series += [code1]
        final = sample['output'][-1]
    elif args.data_type == 'commit':
        if random.random() < args.limit_one_block_prob:
            code_series = random_edit_series(sample['code1'], sample['code2'])[:-1]
        else:
            code_series = [sample['code1']]
        final = sample['code2']
    elif args.data_type == 'submit':
        sample['lang'] = sample['language'].lower()
        code_series = []
        for code1, code2 in zip(sample['submissions'][:-1], sample['submissions'][1:]):
            if random.random() < args.limit_one_block_prob:
                code_series += random_edit_series(code1, code2)[:-1]
            else:
                code_series += [code1]
        final = sample['submissions'][-1]
    else:
        raise ValueError("Data Type {} is not supported.".format(args.data_type))
    histories = [{"type": "code", 'lang': sample['lang'], "code": code} for code in code_series]
    candidates = list(range(len(histories)))
    result = []
    candidate_num = 0
    while len(candidates) > 0 and candidate_num < args.max_per_sample:
        weights = [args.alpha ** c + 1e-6 for c in candidates]
        candidate = random.choices(candidates, weights=weights, k=1)[0]
        candidates = [c for c in candidates if c != candidate]
        current = histories[candidate]
        use_history = random.random() < args.use_history_prob
        use_user = random.random() < args.use_user_prob
        if candidate == 0 and current['code'].strip() == "" and not use_user:
            continue
        if use_history:
            history = histories[:candidate]
            if len(history) == 0:
                continue
        else:
            history = []
        blocks, current_record, future_change = generate_record_and_change(history, current, final)
        if len(future_change) == 0:
            continue
        if use_user:
            if args.data_type == 'aiprogrammer':
                current_record_with_auxiliary = current_record
            elif args.data_type == 'commit':
                current_record_with_auxiliary = current_record + [{"type": "git", "git": sample['git']}]
            elif args.data_type == 'submit':
                current_record_with_auxiliary = current_record + [{"type": "problem", "problem": sample['problems_contexts']}]
            if candidate == 0 and args.data_type == 'commit':
                user = sample['git']
            elif candidate == 0 and args.data_type == 'submit':
                user = "Please generate a correct {} program for the following problem:\n\n{}".format(sample['lang'], sample['problems_contexts'])
            else:
                inst = GenInst.gen([{"record": current_record_with_auxiliary, "change": decorate_code(generate_diff(current['code'], final), "diff")}])
                if inst[0]["output"] is None:
                    continue
                user = inst[0]["output"]
            next_ = {"type": "code", "lang": current['lang'], "code": final}
            current_record += [{"type": "user", "user": user}]
        else:
            judge = GenJudge.gen([{"record": current_record, "change": future_change}])
            if judge[0]["output"] is None:
                continue
            selected_indices = [index for index, value in enumerate(judge[0]["output"]) if value]
            user = ""
            next_ = {"type": "code", "lang": current['lang'], "code": apply_selected_blocks(current['code'], blocks, selected_indices)}
        next_change = decorate_code(generate_diff(current['code'], next_["code"]), "diff")
        chat = GenCht.gen([{"record": current_record, "change": next_change}])
        if chat[0]["output"] is None:
            chat = ""
        else:
            chat = chat[0]["output"]
        result.append({"history": history, "current": current, "user": user, "chat": chat, "next": next_})
        candidate_num += 1
    return result

with concurrent.futures.ThreadPoolExecutor(args.num_proc) as executor:
    futures = [executor.submit(process, item) for item in inputs]
    results = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(future.result())

results = [r for result in results for r in result]
with open(args.output_path, 'w') as f:
    json.dump(list(results), f, indent=4)
