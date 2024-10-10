#TODO: Robust pass@k implementation, only evaluation of greedy decoding now
import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
parser.add_argument("--result_path", type=str, help="Path to the result file")
args = parser.parse_args()

with open(args.dataset_path, 'r') as f:
    dataset = json.load(f)

with open(args.result_path, 'r') as f:
    result = json.load(f)["eval"]

c_base = 0
h_c_base = 0
c_u_base = 0
h_c_u_base = 0

c_extra = 0
h_c_extra = 0
c_u_extra = 0
h_c_u_extra = 0

for sample in dataset:
    if result[sample["task_id"]][0]["base_status"] == "pass":
        if not sample["history"] and not sample["user"]:
            c_base += 1
        elif sample["history"] and not sample["user"]:
            h_c_base += 1
        elif not sample["history"] and sample["user"]:
            c_u_base += 1
        elif sample["history"] and sample["user"]:
            h_c_u_base += 1
    if result[sample["task_id"]][0]["plus_status"] == "pass":
        if not sample["history"] and not sample["user"]:
            c_extra += 1
        elif sample["history"] and not sample["user"]:
            h_c_extra += 1
        elif not sample["history"] and sample["user"]:
            c_u_extra += 1
        elif sample["history"] and sample["user"]:
            h_c_u_extra += 1

print(f"Base Status:")
print(f"  No History, No User: {c_base / 41:.1%}")
print(f"  History, No User: {h_c_base / 41:.1%}")
print(f"  No History, User: {c_u_base / 41:.1%}")
print(f"  History, User: {h_c_u_base / 41:.1%}")
print(f"  Total: {(c_base + h_c_base + c_u_base + h_c_u_base) / 164:.1%}")

print(f"Plus Status:")
print(f"  No History, No User: {c_extra / 41:.1%}")
print(f"  History, No User: {h_c_extra / 41:.1%}")
print(f"  No History, User: {c_u_extra / 41:.1%}")
print(f"  History, User: {h_c_u_extra / 41:.1%}")
print(f"  Total: {(c_extra + h_c_extra + c_u_extra + h_c_u_extra) / 164:.1%}")
