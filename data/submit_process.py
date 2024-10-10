import json
import tqdm
import argparse
from multiprocessing import Pool
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser()
parser.add_argument("--context_path", type=str, help="Path to the Codenet Context dataset")
parser.add_argument("--dataset_path", type=str, help="Path to the Codenet Submit dataset")
parser.add_argument("--identical_path", type=str, help="Path to the Codenet Identical dataset")
parser.add_argument("--output_path", type=str, help="Output Path")
parser.add_argument("--num_proc", type=int, default=32, help="Number of processes")
parser.add_argument("--max_per_problem", type=int, default=50, help="Maximum number of submissions per problem")
parser.add_argument("--max_per_lang_problem", type=int, default=10, help="Maximum number of submissions per language per problem")
parser.add_argument("--internal_similarity_threshold", type=float, default=0.7, help="Internal similarity threshold")
parser.add_argument("--external_similarity_threshold", type=float, default=0.4, help="External similarity threshold")
args = parser.parse_args()

def format_context(context):
    """
    Formats the given context into a text description.

    Args:
        context (dict): The context containing information about the problem.

    Returns:
        str: The formatted text description.

    """
    constraints = context['constraints']
    input_description = context['input_description']
    output_description = context['output_description']
    problem_description = context['problem_description']
    sample_inputs = context['sample_inputs']
    sample_outputs = context['sample_outputs']
    
    # Constructing the text description
    if problem_description:
        text = "Problem Description:\n"
        text += problem_description + "\n\n"
    
    if constraints:
        text += "Constraints:\n"
        text += constraints + "\n\n"
    
    if input_description:
        text += "Input Description:\n"
        text += input_description + "\n\n"
    
    if output_description:
        text += "Output Description:\n"
        text += output_description + "\n\n"
    
    if sample_inputs:
        for i, sample_input, sample_output in zip(range(len(sample_inputs)), sample_inputs, sample_outputs):
            text += f"Sample Input {i + 1}:\n"
            text += sample_input + "\n\n"
            text += f"Sample Output {i + 1}:\n"
            text += sample_output + "\n\n"
    
    return text.strip()

def calculate_rouge_score(fs_tokens):
    """
    Calculates the Rouge score between two sets of tokens.

    Args:
        fs_tokens (tuple): A tuple containing two sets of tokens.

    Returns:
        float: The Rouge score between the two sets of tokens.
    """
    first_tokens, second_tokens = fs_tokens
    return rouge_scorer._score_lcs(first_tokens, second_tokens)

def filter_problem_submissions(problems_submissions, problems_contexts, remove_problem_ids):
    """
    Filters problem submissions based on various criteria including internal and external similarity thresholds.

    Args:
        problems_submissions (dict): A dictionary where keys are problem IDs and values are dictionaries of user submissions.
        problems_contexts (dict): A dictionary where keys are problem IDs and values are the context for each problem.
        remove_problem_ids (set): A set of problem IDs to be removed from consideration.

    Returns:
        list: A list of dictionaries containing filtered submissions with their respective languages and contexts.

    The function performs the following steps:
    1. Initializes a multiprocessing pool and a ROUGE scorer.
    2. Defines a nested function `filter_submissions` to filter individual submissions based on several criteria:
        - Single submission or single line submission.
        - Presence of quadruple newlines.
        - Duplicate submissions.
        - Non-ASCII characters.
        - Internal similarity threshold using ROUGE scores.
    3. Iterates over each problem ID in `problems_submissions`:
        - Skips problem IDs in `remove_problem_ids`.
        - Limits the number of problems and languages per problem based on predefined thresholds.
        - Filters submissions using `filter_submissions`.
        - Checks external similarity threshold using ROUGE scores.
        - Appends valid submissions to the `submit` list.
    """
    pool = Pool(args.num_proc)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    submit = []

    def filter_submissions(submissions):
        """
        Filters a list of submissions based on several criteria.

        Args:
            submissions (list of str): A list of submission strings.

        Returns:
            bool: True if the submissions should be filtered out, False otherwise.

        Criteria for filtering:
        - If there is only one submission.
        - If the last submission contains only one line.
        - If the last submission contains four consecutive newline characters.
        - If there are any duplicate submissions.
        - If any submission contains non-ASCII characters.
        - If the ROUGE score between the final submission and any previous submission is below a specified threshold.
        """
        if len(submissions) == 1 or len(submissions[-1].split("\n")) == 1 or "\n\n\n\n" in submissions[-1]:
            return True
        for s1, s2 in zip(submissions[:-1], submissions[1:]):
            if s1 == s2:
                return True
        for submission in submissions:
            if any(not char.isascii() for char in submission):
                return True
        final_submission_tokens = scorer._tokenizer.tokenize(" ".join(submissions[-1].split()))
        pre_submissions_tokens = [scorer._tokenizer.tokenize(" ".join(submission.split())) for submission in submissions[:-1]]
        pre_final = zip(pre_submissions_tokens, [final_submission_tokens] * len(submissions[:-1]))
        rouge_scores = pool.map(calculate_rouge_score, pre_final)
        rouge_scores = [score.fmeasure for score in rouge_scores]
        if len(rouge_scores) == 0 or max(rouge_scores) < args.internal_similarity_threshold:
            return True
        return False

    for problem_id in tqdm.tqdm(problems_submissions):
        if problem_id in remove_problem_ids:
            continue
        num_problem = 0
        num_lang_problem = {}
        all_final_submissions_tokens = []
        for user_id in problems_submissions[problem_id]:
            if num_problem >= args.max_per_problem:
                break
            lang = problems_submissions[problem_id][user_id]['language']
            if num_lang_problem.get(lang, 0) >= args.max_per_lang_problem:
                continue
            submissions = problems_submissions[problem_id][user_id]["submissions"]
            if not filter_submissions(submissions):
                final_submission_tokens = scorer._tokenizer.tokenize(" ".join(submissions[-1].split()))
                all_final = zip(all_final_submissions_tokens, [final_submission_tokens] * len(all_final_submissions_tokens))
                rouge_scores = pool.map(calculate_rouge_score, all_final)
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if len(rouge_scores) == 0 or max(rouge_scores) < args.external_similarity_threshold:
                    all_final_submissions_tokens.append(final_submission_tokens)
                    num_problem += 1
                    num_lang_problem[lang] = num_lang_problem.get(lang, 0) + 1
                    submit.append({
                        "language": lang,
                        "submissions": submissions,
                        "problems_contexts": format_context(problems_contexts[problem_id]) if problem_id in problems_contexts else ""
                    })
    return submit

with open(args.context_path, 'r') as f:
    codenet_context = json.load(f)

with open(args.dataset_path, 'r') as f:
    codenet_submit = json.load(f)

with open(args.identical_path, 'r') as f:
    identical_problem_clusters = f.read().split("\n")[:-1]

remove_problem_ids = set()
for cluster in identical_problem_clusters:
    problem_ids = cluster.split(",")[1:]
    remove_problem_ids.update(problem_ids)

submit = filter_problem_submissions(codenet_submit, codenet_context, remove_problem_ids)

with open(args.output_path, 'w') as f:
    json.dump(submit, f, indent=4, sort_keys=True)
