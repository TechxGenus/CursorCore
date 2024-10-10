import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import argparse
from gen import AIProgrammer
from generic.utils import data_args, openai_args, get_openai_kwargs, decorate_code

parser = argparse.ArgumentParser()
parser = data_args(parser)
parser = openai_args(parser)
args = parser.parse_args()
openai_kwargs = get_openai_kwargs(args)

with open(args.model_map, 'r') as f:
    model_map = json.load(f)

with open(args.input_path, 'r') as f:
    inputs = json.load(f)

inputs = [{'code': decorate_code(i), "content": i} for i in inputs]

Gen = AIProgrammer(
    model_map=model_map, num_proc=args.num_proc, **openai_kwargs
)

results = Gen.gen(inputs)
results = [result for result in results if result["output"]]

with open(args.output_path, 'w') as f:
    json.dump(results, f, indent=4)
