# Data preprocess

This folder contains preprocessing programs for `Git Commit` and `Online Submit` data.

## Download relevant data

`Online Submit` data needs to be downloaded manually, please refer to [Codenet](https://github.com/IBM/Project_CodeNet) to download.

We clean and translate some of Codenet's question data, which can be found in [Codenet_Context](https://huggingface.co/datasets/TechxGenus/CodeNet_Context).

## Preprocess

Run the following example script to preprocess the data:

```bash
python data/commit.py --languages ruby python javascript shell php java c# c swift typescript c++ go scala rust r --output_path data/commit.json
python data/submit.py --dataset_path <Your Codenet Path> --output_path submit.json
python data/submit_process.py --context_path data/CodeNet_Context.json --dataset_path data/submit.json --identical_path <Your Codenet Path>/derived/duplicates/identical_problem_clusters --output_path data/submit_process.json
```

View the parameters of the scripts can further specify the required language, number of programs, strictness of deduplication, etc.
