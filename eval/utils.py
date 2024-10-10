import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import re
import difflib
import Levenshtein
from generic.special_tokens import *
from .search_and_replace import find_best_match
from generic.utils import decorate_code, extract_changes_lines, generate_locations_changes, generate_search_and_replace

def postprocess_output_wf(current, output):
    """
    Processes the given output string to extract a specific section of text 
    between defined markers and returns the first match found.

    Args:
        current (str): The current string to return in case of an error.
        output (str): The output string to be processed.

    Returns:
        str: The extracted section of text if found, otherwise returns the 
        current string.

    Raises:
        Exception: If an error occurs during processing, the exception is 
        caught and the current string is returned.
    """
    try:
        output = output.split(NEXT_START)[-1].split(NEXT_END)[0]
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        wf = re.findall(pattern, output)
        return wf[0][1]
    except Exception as e:
        print(e)
        return current

def postprocess_output_lc(current, output):
    """
    Post-processes the output by extracting and applying code modifications to the current code.

    Args:
        current (str): The current code as a string.
        output (str): The output containing the modifications.

    Returns:
        str: The updated code after applying the modifications.

    The function expects the `output` to contain code modifications in a specific format:
    - The modifications are enclosed between `NEXT_START` and `NEXT_END` markers.
    - Each modification block follows the pattern: `start_line,end_line\n```<language>\n<code>\n````
    - `start_line` and `end_line` specify the line range in the current code to be replaced by `<code>`.

    If an error occurs during processing, the function prints the exception and returns the original `current` code.
    """
    try:
        output = output.split(NEXT_START)[-1].split(NEXT_END)[0].strip()
        pattern = r"(\d+),(\d+)\n```(.*?)\n([\s\S]*?)\n```"
        lc = re.findall(pattern, output)
        current_lines = current.split("\n")
        for start_line, end_line, _, code in lc[::-1]:
            start_line = int(start_line)
            end_line = int(end_line)
            current_lines = current_lines[:start_line] + code.split("\n") + current_lines[end_line:]
        return "\n".join(current_lines)
    except Exception as e:
        print(e)
        return current

def postprocess_output_sr(current, output):
    """
    Post-processes the output string by extracting and applying search-and-replace operations.

    Args:
        current (str): The current string content to be modified.
        output (str): The output string containing the search-and-replace instructions.

    Returns:
        str: The modified string after applying the search-and-replace operations.

    The function performs the following steps:
    1. Extracts the relevant portion of the output string between NEXT_START and NEXT_END markers.
    2. Finds all code blocks within the extracted portion using a regular expression pattern.
    3. For each code block, splits it into 'before' and 'after' parts using the SEARCH_AND_REPLACE marker.
    4. Finds the best match for the 'before' part in the current string.
    5. Replaces the matched portion in the current string with the 'after' part.
    6. Returns the modified string.

    If an exception occurs during processing, the function prints the exception and returns the original current string.
    """
    try:
        output = output.split(NEXT_START)[-1].split(NEXT_END)[0].strip()
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        sr = re.findall(pattern, output)
        current_lines = current.split("\n")
        for _, before_and_after in sr[::-1]:
            before, after = before_and_after.split("\n" + SEARCH_AND_REPLACE + "\n")
            match_before = find_best_match(before, current)
            current_lines = current_lines[:match_before.start] + after.split("\n") + current_lines[match_before.end:]
        return "\n".join(current_lines)
    except Exception as e:
        print(e)
        return current

def postprocess_output_base(current, output):
    """
    Processes the given output string to extract content between specific markers.

    This function splits the output string using the NEXT_START and NEXT_END markers,
    then uses a regular expression to find content enclosed in triple backticks (```)
    and returns the first match. If an error occurs during processing, the function
    returns the current string.

    Args:
        current (str): The current string to return in case of an error.
        output (str): The output string to be processed.

    Returns:
        str: The extracted content between the markers, or the current string if an error occurs.
    """
    try:
        output = output.split(NEXT_START)[-1].split(NEXT_END)[0]
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        wf = re.findall(pattern, output)
        return wf[0][1]
    except Exception as e:
        print(e)
        return current

def postprocess_output_instruct(current, output):
    """
    Post-processes the given output string by extracting the content between
    specific markers and returning the first code block found.

    Args:
        current (str): The current string to return in case of an error.
        output (str): The output string to be processed.

    Returns:
        str: The extracted code block from the output string. If an error occurs,
             the function returns the 'current' string.
    """
    try:
        output = output.split(NEXT_START)[-1].split(NEXT_END)[0]
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        wf = re.findall(pattern, output)
        return wf[0][1]
    except Exception as e:
        print(e)
        return current

def prepare_input_for_instruct(sample):
    """
    Prepares a conversation input for an instruction-following model based on the provided sample.

    Args:
        sample (dict): A dictionary containing the following keys:
            - "history" (list): A list of dictionaries, each containing:
                - "code" (str): The code from a previous programming process.
                - "lang" (str): The programming language of the code.
            - "current" (dict): A dictionary containing:
                - "code" (str): The current code to be modified.
                - "lang" (str): The programming language of the current code.
            - "user" (str): The user's instruction for modifying the current code.

    Returns:
        list: A list of dictionaries representing the conversation, where each dictionary contains:
            - "role" (str): The role in the conversation, either "user" or "assistant".
            - "content" (str): The content of the message.
    """
    conversation = []
    one_shot = [{"role": "user", "content": "Read the following messages during programming and return the modified code in this format:\n\n<|next_start|>{modified code}<|next_end|>\n\nProgramming process 1:\n```python\na = 1\nb = 2\nc = a + b\n```\n\nCurrent code:\n```python\ni = 1\nb = 2\nc = a + b\n```\n\nUser instruction:\nPlease change variable names."}, {"role": "assistant", "content": "<|next_start|>```python\ni = 1\nj = 2\nk = i + j\n```<|next_end|>"}]
    conversation += one_shot
    prompt = ""
    prompt += "Read the following messages during programming and return the modified code in this format:\n\n<|next_start|>{modified code}<|next_end|>"
    prompt += "\n\n"
    if sample["history"]:
        for i, h in enumerate(sample["history"]):
            prompt += f"Programming process {i + 1}:\n"
            prompt += decorate_code(h["code"], lang=h["lang"]) + "\n\n"
    prompt += "Current code:\n"
    prompt += decorate_code(sample["current"]["code"], lang=sample["current"]["lang"]) + "\n\n"
    if sample["user"]:
        prompt += "User instruction:\n"
        prompt += sample["user"] + "\n\n"
    conversation.append({"role": "user", "content": prompt.strip()})
    return conversation

def prepare_input_for_base(sample):
    """
    Prepares a formatted input string for a base model by combining a prompt, one-shot example, 
    and the provided sample data including history, current code, and user instructions.

    Args:
        sample (dict): A dictionary containing the following keys:
            - "history" (list): A list of dictionaries, each containing:
                - "code" (str): The code snippet from the history.
                - "lang" (str): The programming language of the code snippet.
            - "current" (dict): A dictionary containing:
                - "code" (str): The current code snippet.
                - "lang" (str): The programming language of the current code snippet.
            - "user" (str): The user instruction.

    Returns:
        str: A formatted string that includes the prompt, one-shot example, and the sample data.
    """
    prompt = "Read the following messages during programming and return the modified code in this format:\n\n<|next_start|>{modified code}<|next_end|>\n\n"
    one_shot = "<|messages_start|>Programming process 1:\n```python\na = 1\nb = 2\nc = a + b\n```\n\nCurrent code:\n```python\ni = 1\nb = 2\nc = a + b\n```\n\nUser instruction:\nPlease change variable names.<|messages_end|>\n\n<|next_start|>```python\ni = 1\nj = 2\nk = i + j\n```<|next_end|>\n\n"
    prompt += one_shot
    prompt += "Read the following messages during programming and return the modified code in this format:\n\n<|next_start|>{modified code}<|next_end|>\n\n<|messages_start|>"
    if sample["history"]:
        for i, h in enumerate(sample["history"]):
            prompt += f"Programming process {i + 1}:\n"
            prompt += decorate_code(h["code"], lang=h["lang"]) + "\n\n"
    prompt += "Current code:\n"
    prompt += decorate_code(sample["current"]["code"], lang=sample["current"]["lang"]) + "\n\n"
    if sample["user"]:
        prompt += "User instruction:\n"
        prompt += sample["user"] + "\n\n"
    prompt = prompt.strip() + "<|messages_end|>\n\n"
    return prompt

def prepare_input_for_wf(sample):
    """
    Prepares the input data for workflow processing by formatting the conversation history.

    Args:
        sample (dict): A dictionary containing the following keys:
            - "history" (list): A list of dictionaries representing past messages. Each dictionary contains:
                - "code" (str): The code snippet.
                - "lang" (str): The programming language of the code snippet.
            - "current" (dict): A dictionary representing the current message with the same structure as the history messages.
            - "user" (str): The user's input message.

    Returns:
        list: A list of dictionaries representing the conversation. Each dictionary contains:
            - "role" (str): The role of the message, either "history", "current", or "user".
            - "content" (str): The formatted content of the message.
    """
    conversation = []
    if sample["history"]:
        history_current = sample["history"] + [sample["current"]]
        for m1, m2 in zip(history_current[:-1], history_current[1:]):
            message = decorate_code(m1["code"], m1["lang"])
            conversation.append({"role": "history", "content": message})
    conversation.append({"role": "current", "content": decorate_code(sample["current"]["code"], sample["current"]["lang"])})
    if sample["user"]:
        conversation.append({"role": "user", "content": sample["user"]})
    return conversation

def prepare_input_for_lc(sample):
    """
    Prepares the input data for language model conversation (LC) by processing the sample's history, current code, and user input.

    Args:
        sample (dict): A dictionary containing the following keys:
            - "history" (list): A list of dictionaries representing the history of code changes.
            - "current" (dict): A dictionary representing the current code state with keys:
                - "code" (str): The current code.
                - "lang" (str): The programming language of the current code.
            - "user" (str): The user's input or query.

    Returns:
        list: A list of dictionaries representing the conversation, where each dictionary has:
            - "role" (str): The role in the conversation, either "history", "current", or "user".
            - "content" (str): The content associated with the role, such as code changes or user input.
    """
    conversation = []
    if sample["history"]:
        history_current = sample["history"] + [sample["current"]]
        for m1, m2 in zip(history_current[:-1], history_current[1:]):
            changes_lines = extract_changes_lines(m2["code"], m1["code"])
            locations_changes = generate_locations_changes(m2["code"], m1["code"], m1["lang"], changes_lines)
            conversation.append({"role": "history", "content": locations_changes})
    conversation.append({"role": "current", "content": decorate_code(sample["current"]["code"], sample["current"]["lang"], use_line_num=True)})
    if sample["user"]:
        conversation.append({"role": "user", "content": sample["user"]})
    return conversation

def prepare_input_for_sr(sample):
    """
    Prepares the input for a search and replace (SR) task based on the provided sample.

    Args:
        sample (dict): A dictionary containing the following keys:
            - "history" (list): A list of dictionaries representing the history of code changes.
            - "current" (dict): A dictionary representing the current code state with keys:
                - "code" (str): The current code.
                - "lang" (str): The programming language of the current code.
            - "user" (str, optional): A string representing the user's input or query.

    Returns:
        list: A list of dictionaries representing the conversation for the SR task. Each dictionary contains:
            - "role" (str): The role in the conversation, either "history", "current", or "user".
            - "content" (str): The content associated with the role, such as code changes or user input.
    """
    conversation = []
    if sample["history"]:
        history_current = sample["history"] + [sample["current"]]
        for m1, m2 in zip(history_current[:-1], history_current[1:]):
            changes_lines = extract_changes_lines(m2["code"], m1["code"], unique=True, merge_changes=True)
            changes_lines = [(new, old) for old, new in changes_lines]
            search_and_replace = generate_search_and_replace(m1["code"], m2["code"], m1["lang"], changes_lines)
            conversation.append({"role": "history", "content": search_and_replace})
    conversation.append({"role": "current", "content": decorate_code(sample["current"]["code"], sample["current"]["lang"])})
    if sample["user"]:
        conversation.append({"role": "user", "content": sample["user"]})
    return conversation
