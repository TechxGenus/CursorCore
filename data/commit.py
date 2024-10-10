from datasets import load_dataset
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--languages", help="Supported languages", type=str, nargs="+")
parser.add_argument("--max_per_lang", help="Max samples per language", default=30000, type=int)
parser.add_argument("--output_path", help="Output Path", type=str)
args = parser.parse_args()

commit = []
count_per_lang = {}

def process_sample(sample, count_per_lang, args, commit):
    """
    Processes a sample and updates the commit list based on language constraints.

    Args:
        sample (dict): A dictionary containing sample data with keys "lang", "old_contents", "new_contents", and "subject".
        count_per_lang (dict): A dictionary tracking the count of samples per language.
        args (Namespace): An object containing arguments, specifically:
            - languages (list): A list of languages to include.
            - max_per_lang (int): The maximum number of samples per language.
        commit (list): A list to which the processed sample will be appended if it meets the criteria.

    Returns:
        None
    """
    lang = sample["lang"].lower()
    if lang not in args.languages:
        return
    if lang not in count_per_lang:
        count_per_lang[lang] = 1
    else:
        count_per_lang[lang] += 1
    if count_per_lang[lang] > args.max_per_lang:
        return
    commit.append({
        "code1": sample["old_contents"],
        "code2": sample["new_contents"],
        "lang": lang,
        "git": sample["subject"]
    })

# bigcode/commitpack
# bigcode/commitpackft
# nuprl/EditPackFT
ds = load_dataset("nuprl/EditPackFT-Multi", split="train")
for sample in iter(ds):
    process_sample(sample, count_per_lang, args, commit)

with open(args.output_path, "w") as f:
    json.dump(commit, f, indent=4, sort_keys=True)
