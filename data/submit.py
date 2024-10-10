import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to the Codenet dataset")
parser.add_argument("--output_path", type=str, help="Output Path")
args = parser.parse_args()

metadata_path = os.path.join(args.dataset_path, 'metadata')
data_path = os.path.join(args.dataset_path, 'data')
submit = {}

# Traverse the files in the directory
for file in sorted(os.listdir(metadata_path)):
    file_path = os.path.join(metadata_path, file)
    problem_id = file.split('.')[0]
    if problem_id == "problem_list":
        continue
    submit[problem_id] = {}
    # Load the data
    df = pd.read_csv(file_path)

    # Filter users who use a single language
    grouped_df = df.groupby('user_id')
    language_counts = grouped_df['language'].nunique().reset_index(name='language_count')
    one_lang_users = language_counts[language_counts['language_count'] == 1]
    if len(one_lang_users) == 0:
        continue

    # Filter users who have at least one "Accepted" status
    status_list = grouped_df['status'].unique().reset_index(name='status_list')
    accepted_users = status_list[status_list['status_list'].apply(lambda x: 'Accepted' in x)]
    if len(accepted_users) == 0:
        continue

    # Combine the first two conditions to find the user IDs that meet the criteria
    valid_users = one_lang_users[one_lang_users['user_id'].isin(accepted_users['user_id'])]
    if len(valid_users) == 0:
        continue

    # Filter the submissions of these users
    filtered_submissions = df[df['user_id'].isin(valid_users['user_id'])]

    # Sort by user ID and date
    sorted_submissions = filtered_submissions.sort_values(by=['user_id', 'date'])

    # For each user, find the first submission with "Accepted" status and keep this submission and all previous submissions
    def filter_submissions(sub_df):
        # Find the position of the first record with "Accepted" status
        accepted_index = sub_df[sub_df['status'] == 'Accepted'].index.min()
        if pd.notna(accepted_index):
            # Get the relative position in the sub dataframe
            relative_index = sub_df.index.get_loc(accepted_index)
            # Return the first "Accepted" submission and all previous submissions
            return sub_df.iloc[:relative_index + 1]

    # Apply the filtering logic
    final_submissions = sorted_submissions.groupby('user_id').apply(filter_submissions).reset_index(drop=True)

    for user_id, sub_df in final_submissions.groupby('user_id'):
        # Iterate over each sub dataframe
        submit[problem_id][user_id] = {}
        submit[problem_id][user_id]["submissions"] = []
        language = sub_df['language'].iloc[0]
        submit[problem_id][user_id]['language'] = language
        submissions = sub_df['submission_id'].tolist()
        filename_ext = sub_df['filename_ext'].tolist()
        for submission, ext in zip(submissions, filename_ext):
            submission_path = os.path.join(data_path, problem_id, language, f'{submission}.{ext}')
            with open(submission_path, 'r') as f:
                code = f.read()
            submit[problem_id][user_id]["submissions"].append(code)

with open(args.output_path, 'w') as f:
    json.dump(submit, f, indent=4, sort_keys=True)
