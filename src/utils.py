import random
import difflib
import Levenshtein

def generate_diff(code1, code2, fromfile='', tofile='', n=3):
    """
    Generate a unified diff between two code strings.

    Args:
        code1 (str): The first code string.
        code2 (str): The second code string.
        fromfile (str, optional): The name of the first file. Defaults to ''.
        tofile (str, optional): The name of the second file. Defaults to ''.

    Returns:
        str: The unified diff as a string.
    """
    # Split the code strings into lines
    lines1 = code1.split('\n')
    lines2 = code2.split('\n')

    # Generate unified diff
    diff = difflib.unified_diff(lines1, lines2, fromfile=fromfile, tofile=tofile, n=n, lineterm='')
    output = ""

    for line in diff:
        output += line + "\n"

    if fromfile == '' and tofile == '':
        output = "\n".join(output.split("\n")[2:])

    return output[:-1]
    
def if_continuous_modify(code1, code2, code3):
    """
    Check if code3 is a continuous modification of code1 and code2.

    Args:
        code1 (str): The first code string.
        code2 (str): The second code string.
        code3 (str): The third code string.

    Returns:
        bool: True if code3 is a continuous modification of code1 and code2, False otherwise.
    """
    # Calculate the Levenshtein distance between code1 and code2
    dist1 = Levenshtein.distance(code1, code2)
    # Calculate the Levenshtein distance between code2 and code3
    dist2 = Levenshtein.distance(code2, code3)
    # Calculate the Levenshtein distance between code1 and code3
    dist3 = Levenshtein.distance(code1, code3)

    # Check if code3 is a continuous modification of code1 and code2
    if dist3 == dist1 + dist2:
        return True
    else:
        return False
    
def blockwise_if_continuous_modify(code1, code2, code3):
    """
    Check if code3 is a continuous modification of code1 and code2.

    Args:
        code1 (str): The first code string.
        code2 (str): The second code string.
        code3 (str): The third code string.

    Returns:
        bool: True if code3 is a continuous modification of code1 and code2, False otherwise.
    """
    # Calculate the Levenshtein distance between code1 and code2
    dist1 = Levenshtein.distance(code1, code2)
    # Calculate the Levenshtein distance between code2 and code3
    dist2 = Levenshtein.distance(code2, code3)
    # Calculate the Levenshtein distance between code1 and code3
    dist3 = Levenshtein.distance(code1, code3)

    past_diff_blocks = generate_diff_blocks(code1, code2)
    new_diff_blocks = generate_diff_blocks(code1, code3)
    
    # Check if code3 is a continuous modification of code1 and code2
    if dist3 == dist1 + dist2 and len(past_diff_blocks) == len(new_diff_blocks):
        return True
    else:
        return False

def generate_diff_blocks(original, modified):
    """
    Generate diff blocks between two strings.

    Args:
        original (str): The original string.
        modified (str): The modified string.

    Returns:
        list: A list of tuples, where each tuple contains a block of modified lines and the line number in the original string where the block starts.
    """
    # Use difflib's ndiff to find differences
    differ = difflib.Differ()
    diff = list(differ.compare(original.split('\n'), modified.split('\n')))
    
    # store all modified blocks
    blocks = []
    current_block = []
    
    # track the current line number
    orig_line_no = 0
    block_line_len = 0

    # Traverse the diff results into chunks
    for line in diff:
        if line.startswith('  '):
            # If the current block has content and an unmodified line is encountered, save the current block and reset
            if current_block:
                blocks.append((current_block, orig_line_no - block_line_len))
                current_block = []
                block_line_len = 0
            orig_line_no += 1
        elif line.startswith('- '):
            current_block.append(line)
            orig_line_no += 1
            block_line_len += 1
        else:
            current_block.append(line)
    
    # Make sure the last chunk is added
    if current_block:
        blocks.append((current_block, orig_line_no - block_line_len))
    
    return blocks

def apply_selected_blocks(original, blocks, selected_indices):
    """
    Apply selected blocks to the original code.

    Args:
        original (str): The original code as a string.
        blocks (list): A list of code blocks.
        selected_indices (list): A list of indices representing the selected blocks.

    Returns:
        str: The modified code after applying the selected blocks.
    """
    # Split the original code by lines
    original_lines = original.split('\n')
    
    # Adjust code based on selected block
    offset = 0
    for index in selected_indices:
        block, start_line = blocks[index]
        # Iterate through each row in the block
        delete_offset = 0
        for line in block:
            if line.startswith('- '):
                del original_lines[start_line + offset]
                delete_offset -= 1
            elif line.startswith('+ '):
                original_lines.insert(start_line + offset, line[2:])
                offset += 1
        offset += delete_offset
        delete_offset = 0
    
    return '\n'.join(original_lines)

# TODO: accelerate the process
def random_edit_series(code1, code2):
    """
    Generates a series of randomly edited code versions between two given code snippets.

    Args:
        code1 (str): The original code snippet.
        code2 (str): The modified code snippet.

    Returns:
        list: A list of code snippets representing the series of randomly edited versions
              between the original and modified code snippets.
    """
    blocks = generate_diff_blocks(code1, code2)
    random_block_order = list(range(len(blocks)))
    # random.seed(42)
    random.shuffle(random_block_order)
    code_series = []
    for i in range(len(blocks) + 1):
        code_series.append(apply_selected_blocks(code1, blocks, sorted(random_block_order[:i])))
    return code_series

def sequential_edit_series(code1, code2):
    """
    Generate a series of code snippets by sequentially applying selected diff blocks.

    Args:
        code1 (str): The original code.
        code2 (str): The modified code.

    Returns:
        list: A list of code snippets, each representing the result of applying a selected diff block.
    """
    blocks = generate_diff_blocks(code1, code2)
    code_series = []
    for i in range(len(blocks)):
        code_series.append(apply_selected_blocks(code1, blocks, [i]))
    return code_series
