import re

def extract_code_blocks(markdown):
    """
    Extracts code blocks from a given markdown string.

    Args:
        markdown (str): The markdown string to extract code blocks from.

    Returns:
        list: A list of code blocks extracted from the markdown string.
    """
    # Define a regular expression to match code blocks. The (?s) allows '.' to match all characters including newlines.
    code_block_pattern = re.compile(r'```(.*?)\n(.*?)```', re.DOTALL)

    # Store the extracted code blocks
    code_blocks = []

    # Use finditer to iterate over all matches of code blocks
    for block in code_block_pattern.finditer(markdown):
        # Get the content of the code block
        code_content = block.group(2)

        # Remove any additional indents (when the code block is in lists or other indented structures)
        lines = code_content.split('\n')
        code_blocks.append('\n'.join(line[len(lines[-1]):] for line in lines[:-1]))

    return code_blocks
