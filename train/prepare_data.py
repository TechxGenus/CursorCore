import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import argparse
from generic.special_tokens import *
from generic.utils import decorate_code, extract_changes_lines, generate_locations_changes, generate_search_and_replace

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Input Path")
parser.add_argument("--output_path", type=str, help="Output Path")
parser.add_argument("--format_type", type=str, default="wf", help="Format Type for data")
args = parser.parse_args()

def format_data(data):
    """
    Formats the given data based on the specified format type in args.

    Args:
        data (list): A list of dictionaries containing the data to be formatted. Each dictionary represents a sample and contains the following keys:
            - "history" (list): A list of dictionaries representing the history of messages. Each dictionary contains:
                - "code" (str): The code snippet.
                - "lang" (str): The programming language of the code snippet.
            - "current" (dict): A dictionary representing the current message with keys:
                - "code" (str): The current code snippet.
                - "lang" (str): The programming language of the current code snippet.
            - "loc" (int or tuple): The location(s) in the current code snippet where the target(s) should be inserted.
            - "user" (str): The user's message.
            - "next" (dict): A dictionary representing the next message with keys:
                - "code" (str): The next code snippet.
                - "lang" (str): The programming language of the next code snippet.
            - "chat" (str): Additional chat message from the assistant.

    Returns:
        list: A list of formatted conversations. Each conversation is a list of dictionaries with the following keys:
            - "role" (str): The role of the message ("history", "current", "user", "assistant").
            - "content" (str): The content of the message, which may include decorated code or location changes.

    Raises:
        ValueError: If the format type specified in args is invalid.
    """
    formatted_data = []
    if args.format_type == "wf":
        for sample in data:
            conversation = []
            if sample["history"]:
                for message in sample["history"]:
                    conversation.append({"role": "history", "content": decorate_code(message["code"], message["lang"])})
            current_loc = sample["current"]["code"]
            if sample["loc"]:
                if type(sample["loc"]) == int:
                    current_loc = current_loc[:sample["loc"]] + TARGET + current_loc[sample["loc"]:]
                else:
                    start, end = sample["loc"]
                    current_loc = current_loc[:end] + TARGET_END + current_loc[end:]
                    current_loc = current_loc[:start] + TARGET_START + current_loc[start:]
            conversation.append({"role": "current", "content": decorate_code(current_loc, sample["current"]["lang"])})
            if sample["user"]:
                conversation.append({"role": "user", "content": sample["user"]})
            assistant = ""
            assistant += NEXT_START + decorate_code(sample["next"]["code"], sample["next"]["lang"]) + NEXT_END
            if sample["chat"]:
                assistant += "\n" + sample["chat"]
            conversation.append({"role": "assistant", "content": assistant})
            formatted_data.append(conversation)
    elif args.format_type == "lc":
        for sample in data:
            conversation = []
            if sample["history"]:
                history_current = sample["history"] + [sample["current"]]
                for m1, m2 in zip(history_current[:-1], history_current[1:]):
                    # In H, we record the modified position of the subsequent code snippet and the modified content of the previous code snippet
                    # in A, we record the modified position of the previous code snippet and the modified content of the subsequent code snippet
                    changes_lines = extract_changes_lines(m2["code"], m1["code"])
                    locations_changes = generate_locations_changes(m2["code"], m1["code"], m1["lang"], changes_lines)
                    conversation.append({"role": "history", "content": locations_changes})
            current_loc = sample["current"]["code"]
            if sample["loc"]:
                if type(sample["loc"]) == int:
                    current_loc = current_loc[:sample["loc"]] + TARGET + current_loc[sample["loc"]:]
                else:
                    start, end = sample["loc"]
                    current_loc = current_loc[:end] + TARGET_END + current_loc[end:]
                    current_loc = current_loc[:start] + TARGET_START + current_loc[start:]
            # we add line numbers for assistant to understand the location
            conversation.append({"role": "current", "content": decorate_code(current_loc, sample["current"]["lang"], use_line_num=True)})
            if sample["user"]:
                conversation.append({"role": "user", "content": sample["user"]})
            assistant = ""
            changes_lines = extract_changes_lines(sample["current"]["code"], sample["next"]["code"])
            locations_changes = generate_locations_changes(sample["current"]["code"], sample["next"]["code"], sample["next"]["lang"], changes_lines)
            assistant += NEXT_START + locations_changes + NEXT_END
            if sample["chat"]:
                assistant += "\n" + sample["chat"]
            conversation.append({"role": "assistant", "content": assistant})
            formatted_data.append(conversation)
    elif args.format_type == "sr":
        for sample in data:
            conversation = []
            if sample["history"]:
                history_current = sample["history"] + [sample["current"]]
                for m1, m2 in zip(history_current[:-1], history_current[1:]):
                    # We ensure that the searched content matches exactly and there are no duplicate paragraphs
                    changes_lines = extract_changes_lines(m2["code"], m1["code"], unique=True, merge_changes=True)
                    changes_lines = [(new, old) for old, new in changes_lines]
                    search_and_replace = generate_search_and_replace(m1["code"], m2["code"], m1["lang"], changes_lines)
                    conversation.append({"role": "history", "content": search_and_replace})
            current_loc = sample["current"]["code"]
            if sample["loc"]:
                if type(sample["loc"]) == int:
                    current_loc = current_loc[:sample["loc"]] + TARGET + current_loc[sample["loc"]:]
                else:
                    start, end = sample["loc"]
                    current_loc = current_loc[:end] + TARGET_END + current_loc[end:]
                    current_loc = current_loc[:start] + TARGET_START + current_loc[start:]
            conversation.append({"role": "current", "content": decorate_code(current_loc, sample["current"]["lang"])})
            if sample["user"]:
                conversation.append({"role": "user", "content": sample["user"]})
            assistant = ""
            changes_lines = extract_changes_lines(sample["current"]["code"], sample["next"]["code"], unique=True, merge_changes=True)
            search_and_replace = generate_search_and_replace(sample["current"]["code"], sample["next"]["code"], sample["next"]["lang"], changes_lines)
            assistant += NEXT_START + search_and_replace + NEXT_END
            if sample["chat"]:
                assistant += "\n" + sample["chat"]
            conversation.append({"role": "assistant", "content": assistant})
            formatted_data.append(conversation)
    else:
        raise ValueError(f"Invalid format type: {args.format_type}")
    return formatted_data

with open(args.input_path, "r") as f:
    input_data = json.load(f)

output_data = format_data(input_data)

with open(args.output_path, "w") as f:
    json.dump(output_data, f, indent=4)
