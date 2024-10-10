import json
import random
import argparse
import Levenshtein

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Input Path")
parser.add_argument("--output_path", type=str, help="Output Path")
parser.add_argument("--use_loc_prob", type=float, default=0.75, help="Probability of using LOC")
parser.add_argument("--loc_cursor_prob", type=float, default=0.5, help="Probability of using cursor, using range otherwise")
parser.add_argument("--loc_noisy_prob", type=float, default=0.5, help="Probability of using noisy range")
parser.add_argument("--loc_noisy_length", type=int, default=32, help="Length of noisy range")
parser.add_argument("--loc_all_noisy_prob", type=float, default=0.05, help="Probability of using noisy range for all")
args = parser.parse_args()

def get_changes(s1, s2):
    """
    Compute the changes needed to transform string s1 into string s2 using Levenshtein distance.

    Args:
        s1 (str): The original string.
        s2 (str): The target string.

    Returns:
        list of tuple: A list of tuples where each tuple contains the start and end indices of the changes in s1.
                       Each tuple represents a range of indices in s1 that need to be changed to match s2.
    """
    edit_ops = Levenshtein.opcodes(s1, s2)
    changes = []
    for op, i1, i2, _, _ in edit_ops:
        if op != 'equal':
            if changes and changes[-1][1] == i1:
                changes[-1] = (changes[-1][0], i2)
            else:
                changes.append((i1, i2))
    return changes

def sample_from_boundaries(left, right, max_length, noisy_length=0):
    """
    Samples a start and end point within specified boundaries, with optional noise.

    Args:
        left (int): The left boundary for sampling the start point.
        right (int): The right boundary for sampling the end point.
        max_length (int): The maximum allowable length for the end point.
        noisy_length (int, optional): The amount of noise to add to the boundaries. Defaults to 0.

    Returns:
        tuple: A tuple containing the sampled start and end points (start, end).
    """
    start = random.choice(list(range(max(0, left - noisy_length), left + 1)))
    end = random.choice(list(range(right, min(right + noisy_length, max_length) + 1)))
    return start, end

def sample_from_ranges(ranges):
    """
    Selects a random number from a set of ranges.

    Args:
        ranges (list of tuple): A list of tuples where each tuple contains two integers (start, end) 
                                representing the inclusive range from start to end.

    Returns:
        int: A randomly selected number from the combined ranges.

    Raises:
        ValueError: If the ranges list is empty or if no numbers can be generated from the given ranges.
    """
    numbers = set()
    for start, end in ranges:
        numbers.update(range(start, end + 1))
    return random.choice(list(numbers))

def post_current(current, next_):
    """
    Determines the cursor position or range based on the changes between the current and next states.

    Args:
        current (str): The current state.
        next_ (str): The next state.

    Returns:
        Union[int, Tuple[int, int], None]: 
            - An integer representing a single cursor position.
            - A tuple of two integers representing a range (left, right).
            - None if no cursor position or range is determined.
    """
    if random.random() < args.use_loc_prob:
        changes = get_changes(current, next_)
        if not changes:
            changes = [(0, len(current))]
        if random.random() < args.loc_cursor_prob:
            if random.random() < args.loc_all_noisy_prob:
                cursor = sample_from_ranges([(0, len(current))])
            else:
                if random.random() < args.loc_noisy_prob:
                    cursor = sample_from_ranges([random.choice(changes)])
                else:
                    cursor = random.choice(random.choice(changes))
            return cursor
        else:
            if random.random() < args.loc_all_noisy_prob:
                left, right = sample_from_ranges([(0, len(current))]), sample_from_ranges([(0, len(current))])
                if left > right:
                    left, right = right, left
            else:
                if random.random() < args.loc_noisy_prob:
                    noisy_length = args.loc_noisy_length
                else:
                    noisy_length = 0
                left, right = sample_from_boundaries(changes[0][0], changes[-1][1], len(current), noisy_length)
            return left, right
    else:
        return None

with open(args.input_path, "r") as f:
    data = json.load(f)

for sample in data:
    loc = post_current(sample["current"]["code"], sample["next"]["code"])
    # TODO: Unify variable names
    sample["loc"] = loc

with open(args.output_path, "w") as f:
    json.dump(data, f, indent=4)
