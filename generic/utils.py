import difflib
from .special_tokens import *

def data_args(parser):
    """
    Adds data arguments to the given argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the arguments to.

    Returns:
        argparse.ArgumentParser: The updated argument parser.
    """
    parser.add_argument("--model_map", type=str, help="Model name, base and port")
    parser.add_argument("--input_path", type=str, help="Input Path")
    parser.add_argument("--output_path", type=str, help="Output Path")
    parser.add_argument("--num_proc", type=int, default=512, help="Number of processes")
    return parser

def openai_args(parser):
    """
    Adds OpenAI arguments to the given argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the arguments to.

    Returns:
        argparse.ArgumentParser: The updated argument parser.
    """
    parser.add_argument("--max_tokens", type=int, default=3072, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0, help="Temperature")
    parser.add_argument("--presence_penalty", type=float, default=0, help="Temperature")
    return parser

def get_openai_kwargs(args):
    """
    Get OpenAI keyword arguments from the given arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        dict: The OpenAI keyword arguments.
    """
    openai_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
    }
    return openai_kwargs

def decorate_code(code, lang="", use_line_num=False, start_line=None, end_line=None):
    """
    Decorates the given code with markdown syntax.

    Args:
        code (str): The code to be decorated.
        lang (str, optional): The language identifier for syntax highlighting. Defaults to "".
        use_line_num (bool, optional): Whether to include line numbers. Defaults to False.
        start_line (int, optional): The starting line number. Defaults to None.
        end_line (int, optional): The ending line number. Defaults to None.

    Returns:
        str: The decorated code.

    """
    decorate = "```"
    if lang:
        decorate += lang.lower()
    decorate += "\n"
    code_lines = code.split("\n")
    if start_line is None:
        start_line = 0
    if end_line is None:
        end_line = len(code_lines)
    if start_line != 0:
        decorate += "...\n"
    if use_line_num:
        decorate += "\n".join([f"{i + 1 + start_line} {line}" for i, line in enumerate(code_lines[start_line: end_line])])
    else:
        decorate += "\n".join(code_lines[start_line: end_line])
    if end_line != len(code_lines):
        decorate += "\n..."
    decorate += "\n```"
    return decorate

def generate_locations_changes(code1, code2, lang, changes_lines):
    """
    Generates a string representing the changes between two versions of code with specified line ranges.

    Args:
        code1 (str): The original version of the code.
        code2 (str): The modified version of the code.
        lang (str): The programming language of the code (used for syntax highlighting).
        changes_lines (list of tuples): A list of tuples where each tuple contains two pairs of integers.
            Each tuple represents the line ranges in the format ((old_start, old_end), (new_start, new_end)).

    Returns:
        str: A formatted string that shows the changes between the two code versions with syntax highlighting.
    """
    code1_lines = code1.split('\n')
    code2_lines = code2.split('\n')
    locations_changes = []
    for c in changes_lines:
        (old_start, old_end), (new_start, new_end) = c
        next_code = '\n'.join(code2_lines[new_start: new_end])
        locations_changes.append(f"{old_start},{old_end}\n```{lang}\n{next_code}\n```")
    return "\n".join(locations_changes)

def generate_search_and_replace(code1, code2, lang, changes_lines, sep_token=SEARCH_AND_REPLACE):
    """
    Generates a search and replace string for code changes between two versions of code.

    Args:
        code1 (str): The original version of the code.
        code2 (str): The modified version of the code.
        lang (str): The programming language of the code (used for syntax highlighting).
        changes_lines (list of tuples): A list of tuples where each tuple contains two pairs of integers.
            Each pair represents the start and end line numbers of the changes in the original and modified code respectively.
        sep_token (str, optional): The token used to separate the search and replace sections in the output. Defaults to SEARCH_AND_REPLACE.

    Returns:
        str: A formatted string containing the search and replace sections for the specified changes, 
             with syntax highlighting for the specified programming language.
    """
    code1_lines = code1.split('\n')
    code2_lines = code2.split('\n')
    search_and_replace = []
    for i in range(len(changes_lines)):
        (old_start, old_end), (new_start, new_end) = changes_lines[i]
        search = '\n'.join(code1_lines[old_start: old_end])
        replace = '\n'.join(code2_lines[new_start: new_end])
        search_and_replace.append(f"```{lang}\n{search}\n{sep_token}\n{replace}\n```")
    return "\n".join(search_and_replace)

def extract_changes_lines(code1, code2, fromfile='', tofile='', unique=False, merge_changes=False, merge_threshold=2):
    """
    Extracts the lines that have changed between two versions of code.
    Args:
        code1 (str): The original version of the code.
        code2 (str): The modified version of the code.
        fromfile (str, optional): The name of the original file. Defaults to ''.
        tofile (str, optional): The name of the modified file. Defaults to ''.
        unique (bool, optional): If True, ensures that the changes are unique. Defaults to False.
        merge_changes (bool, optional): If True, merges neighboring changes. Defaults to False.
        merge_threshold (int, optional): The threshold for merging neighboring changes. Defaults to 2.
    Returns:
        list: A list of tuples, where each tuple contains two tuples representing the start and end lines of changes 
              in the original and modified code respectively.
    """
    # Split the code into lines
    lines1 = code1.split('\n')
    lines2 = code2.split('\n')
    
    # Create a unified diff
    diff = list(difflib.unified_diff(lines1, lines2, fromfile=fromfile, tofile=tofile, n=0, lineterm=''))[2:]
    
    # Process the diff to extract changes and their contexts from both files
    changes_lines = []
    current_old_start_line = None
    current_new_start_line = None
    current_old_end_line = 0
    current_new_end_line = 0
    
    for line in diff:
        if line.startswith('@@'):
            # Start of a new change block, flush the previous block if it exists
            if current_old_start_line or current_new_start_line:
                if current_old_end_line > current_old_start_line:
                    current_old_start_line -= 1
                    current_old_end_line -= 1
                if current_new_end_line > current_new_start_line:
                    current_new_start_line -= 1
                    current_new_end_line -= 1
                changes_lines.append(((current_old_start_line, current_old_end_line),
                                             (current_new_start_line, current_new_end_line)))

            # Extract line numbers from the hunk information
            parts = line.split()
            old_info, new_info = parts[1], parts[2]
            current_old_start_line = int(old_info.split(',')[0][1:])
            current_new_start_line = int(new_info.split(',')[0][1:])
            current_old_end_line = current_old_start_line
            current_new_end_line = current_new_start_line
        elif line.startswith('-'):
            # Line removed from the old file
            current_old_end_line += 1
        elif line.startswith('+'):
            # Line added to the new file
            current_new_end_line += 1
        elif line.startswith(' '):
            assert False, "No lines should be unchanged in a unified diff with context lines number 0"
    
    # Append the last block if it exists
    if current_old_start_line or current_new_start_line:
        if current_old_end_line > current_old_start_line:
            current_old_start_line -= 1
            current_old_end_line -= 1
        if current_new_end_line > current_new_start_line:
            current_new_start_line -= 1
            current_new_end_line -= 1
        changes_lines.append(((current_old_start_line, current_old_end_line),
                                     (current_new_start_line, current_new_end_line)))
    changes_lines = filter_changes(changes_lines, lines1, lines2)
    if unique:
        changes_lines = unique_changes(changes_lines, lines1)
    if merge_changes:
        changes_lines = merge_neighbor_change(changes_lines, merge_threshold)
    return changes_lines

def filter_changes(changes_lines, lines1, lines2):
    """
    Filters out changes that are only due to differences in indentation or whitespace.

    Args:
        changes_lines (list of tuples): A list of tuples where each tuple contains two tuples.
                                        The first tuple represents the start and end indices of the change in `lines1`.
                                        The second tuple represents the start and end indices of the change in `lines2`.
        lines1 (list of str): The original list of lines.
        lines2 (list of str): The modified list of lines.

    Returns:
        list of tuples: A list of tuples containing the changes that are not purely due to differences in indentation or whitespace.
    """
    filtered_changes_lines = []
    if len(changes_lines) == 0:
        return filtered_changes_lines
    for old_change, new_change in changes_lines:
        old_start, old_end = old_change
        new_start, new_end = new_change
        indent1_max = max(len(line) - len(line.lstrip()) for line in lines1)
        indent2_max = max(len(line) - len(line.lstrip()) for line in lines2)
        before = " ".join(" ".join(lines1[old_start: old_end]).split())
        after = " ".join(" ".join(lines2[new_start: new_end]).split())
        if before != after or indent1_max != indent2_max:
            filtered_changes_lines.append((old_change, new_change))
    return filtered_changes_lines

def unique_changes(changes_lines, lines1):
    """
    Identifies unique changes in a list of changes and expands them based on the context of the original lines.

    Args:
        changes_lines (list of tuples): A list of tuples where each tuple contains two sub-tuples. 
            The first sub-tuple represents the old change with start and end indices, 
            and the second sub-tuple represents the new change with start and end indices.
        lines1 (list): A list of lines from the original content.

    Returns:
        list of tuples: A list of tuples where each tuple contains two sub-tuples. 
            The first sub-tuple represents the expanded old change with updated start and end indices, 
            and the second sub-tuple represents the expanded new change with updated start and end indices.

    Raises:
        AssertionError: If the left or right expansion values are negative, indicating an invalid expansion.
    """
    unique_changes_lines = []
    if len(changes_lines) == 0:
        return unique_changes_lines
    for old_change, new_change in changes_lines:
        left_expansion, right_expansion = find_unique_sublist(lines1, old_change[0], old_change[1])
        assert left_expansion >= 0 and right_expansion >= 0, "Invalid expansion"
        unique_changes_lines.append(((old_change[0] - left_expansion, old_change[1] + right_expansion), (new_change[0] - left_expansion, new_change[1] + right_expansion)))
    return unique_changes_lines

# TODO: accelerate
def find_unique_sublist(b, a1, a2):
    """
    Finds the smallest extension of the sublist `b[a1:a2]` that is unique within the list `b`.

    Args:
        b (list): The list in which to find the unique sublist.
        a1 (int): The starting index of the sublist.
        a2 (int): The ending index of the sublist.

    Returns:
        tuple: A tuple (x, y) where `x` is the minimum number of elements to extend the sublist
               at the beginning, and `y` is the minimum number of elements to extend the sublist
               at the end to make it unique within `b`. If no unique sublist is found, returns (-1, -1).
    """
    if not "\n".join(b).strip():
        return 0, 0

    def is_unique_sublist(b, sublist):
        """
        Check if a sublist appears exactly once in a list.

        Args:
            b (list): The main list in which to search for the sublist.
            sublist (list): The sublist to search for within the main list.

        Returns:
            bool: True if the sublist appears exactly once in the main list, False otherwise.

        Raises:
            AssertionError: If the sublist appears more than once in the main list.
        """
        if not "\n".join(sublist).strip():
            return False
        count = 0
        for i in range(len(b) - len(sublist) + 1):
            if b[i:i + len(sublist)] == sublist:
                count += 1
            if count > 1:
                return False
        assert count == 1, "Invalid count"
        return count == 1

    for extend_length in range(len(b) - a2 + a1 + 1):
        for window in range(extend_length + 1):
            current_sublist = b[a1 - extend_length + window:a2 + window]
            if is_unique_sublist(b, current_sublist):
                return min(a1, extend_length - window), min(len(b) - a2, window)
    return -1, -1

def merge_neighbor_change(changes_lines, merge_threshold=2):
    """
    Merges neighboring changes in a list of change line ranges based on a specified threshold.

    Args:
        changes_lines (list of tuples): A list of tuples where each tuple contains two tuples representing 
                                        the old and new change line ranges respectively. 
                                        Example: [((old_start, old_end), (new_start, new_end)), ...]
        merge_threshold (int, optional): The maximum allowed distance between neighboring changes to be merged. 
                                         Defaults to 2.

    Returns:
        list of tuples: A list of merged change line ranges.
    """
    merged_changes_lines = []
    if len(changes_lines) == 0:
        return merged_changes_lines
    for i, (old_change, new_change) in enumerate(changes_lines):
        if i == 0:
            merged_changes_lines.append((old_change, new_change))
        else:
            last_old_change, last_new_change = merged_changes_lines[-1]
            if old_change[0] - last_old_change[1] <= merge_threshold and new_change[0] - last_new_change[1] <= merge_threshold:
                merged_changes_lines[-1] = (min(last_old_change[0], old_change[0]), max(last_old_change[1], old_change[1])), (min(last_new_change[0], new_change[0]), max(last_new_change[1], new_change[1]))
            else:
                merged_changes_lines.append((old_change, new_change))
    return merged_changes_lines
