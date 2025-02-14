#TODO: Robust pass@k implementation, only evaluation of greedy decoding now
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
parser.add_argument("--result_path", type=str, help="Path to the result file")
args = parser.parse_args()

with open(args.dataset_path, "r") as f:
    dataset = json.load(f)

with open(args.result_path, "r") as f:
    result = json.load(f)["eval"]

# -------------------------------------------------
# 1. Count how many samples are in each category
# -------------------------------------------------
total_nh_nu = 0  # No history, no user
total_h_nu = 0  # History, no user
total_nh_u = 0  # No history, user
total_h_u = 0  # History, user

for sample in dataset:
    if not sample["history"] and not sample["user"]:
        total_nh_nu += 1
    elif sample["history"] and not sample["user"]:
        total_h_nu += 1
    elif not sample["history"] and sample["user"]:
        total_nh_u += 1
    else:
        total_h_u += 1

total_all = len(dataset)

# -------------------------------------------------
# 2. Initialize counters for passes in each category
# -------------------------------------------------
c_base = 0  # base pass: no history, no user
h_c_base = 0  # base pass: history, no user
c_u_base = 0  # base pass: no history, user
h_c_u_base = 0  # base pass: history, user

c_extra = 0  # plus pass: no history, no user
h_c_extra = 0  # plus pass: history, no user
c_u_extra = 0  # plus pass: no history, user
h_c_u_extra = 0  # plus pass: history, user

# -------------------------------------------------
# 3. Fill in counters by checking pass status
# -------------------------------------------------
for sample in dataset:
    base_status = result[sample["task_id"]][0]["base_status"]
    plus_status = result[sample["task_id"]][0]["plus_status"]

    # Base pass checks
    if base_status == "pass":
        if not sample["history"] and not sample["user"]:
            c_base += 1
        elif sample["history"] and not sample["user"]:
            h_c_base += 1
        elif not sample["history"] and sample["user"]:
            c_u_base += 1
        else:
            h_c_u_base += 1

    # Plus pass checks
    if plus_status == "pass":
        if not sample["history"] and not sample["user"]:
            c_extra += 1
        elif sample["history"] and not sample["user"]:
            h_c_extra += 1
        elif not sample["history"] and sample["user"]:
            c_u_extra += 1
        else:
            h_c_u_extra += 1


# -------------------------------------------------
# 4. Helper function to safely handle divisions
# -------------------------------------------------
def ratio_str(numerator, denominator):
    """Return 'count/denominator (xx.x%)', handling zero denominator."""
    if denominator == 0:
        return f"{numerator}/0 (N/A)"
    else:
        return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


# -------------------------------------------------
# 5. Print results for Base
# -------------------------------------------------
print("Base Status:")
print(f"  No History, No User: {ratio_str(c_base, total_nh_nu)}")
print(f"  History, No User:   {ratio_str(h_c_base, total_h_nu)}")
print(f"  No History, User:   {ratio_str(c_u_base, total_nh_u)}")
print(f"  History, User:      {ratio_str(h_c_u_base, total_h_u)}")
print(
    f"  Total:              {ratio_str(c_base + h_c_base + c_u_base + h_c_u_base, total_all)}"
)

# -------------------------------------------------
# 6. Print results for Plus
# -------------------------------------------------
print("\nPlus Status:")
print(f"  No History, No User: {ratio_str(c_extra, total_nh_nu)}")
print(f"  History, No User:   {ratio_str(h_c_extra, total_h_nu)}")
print(f"  No History, User:   {ratio_str(c_u_extra, total_nh_u)}")
print(f"  History, User:      {ratio_str(h_c_u_extra, total_h_u)}")
print(
    f"  Total:              {ratio_str(c_extra + h_c_extra + c_u_extra + h_c_u_extra, total_all)}"
)
