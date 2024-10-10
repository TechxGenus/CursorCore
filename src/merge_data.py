import json
import argparse

def merge_json_files(output_file, *input_files):
    """
    Merge multiple JSON files into a single file.

    Args:
        output_file (str): The path to the output file where the merged data will be written.
        *input_files (str): Variable number of input file paths to be merged.

    Raises:
        FileNotFoundError: If any of the input files are not found.
        JSONDecodeError: If any of the input files cannot be parsed as JSON.

    Returns:
        None
    """
    merged_data = []
    for file in input_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {file} does not contain a list. Skipping.")
        except FileNotFoundError as e:
            print(f"Error reading {file}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing {file} as JSON: {e}")
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Merged data written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

parser = argparse.ArgumentParser(description='Merge multiple JSON files into one.')
parser.add_argument('output', type=str, help='The output file to write the merged data to.')
parser.add_argument('inputs', nargs='+', type=str, help='The input JSON files to merge.')

args = parser.parse_args()
merge_json_files(args.output, *args.inputs)
