# Programming-Instruct

This folder is the code for our data collection pipeline.

## Inference service for LLMs

We generate data by prompting different LLMs. Currently, the OpenAI interface has been standardized, and almost all Inference frameworks support OpenAI compatible servers. Therefore, we uniformly use the OpenAI interface to generate.

Example script to deploy `deepseek-coder-6.7b-instruct` using `sglang`:

```bash
python -m sglang.launch_server --model-path deepseek-ai/deepseek-coder-6.7b-instruct --port 10086
```

Example script to deploy `deepseek-coder-6.7b-instruct` using `vllm`:

```bash
python -m vllm.entrypoints.openai.api_server --port 10086 --model deepseek-ai/deepseek-coder-6.7b-instruct --enable-prefix-caching
```

We define the model inference service parameters in `model_map.json`. An example configuration is as follows:

```json
{
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "base": "http://127.0.0.1:10086/v1",
        "api": "sk-xxx"
    }
}
```

## AI programmer

For each code snippet, we use LLMs to generate the corresponding coding history. Its input file is a list of code snippets. Examples are as follows:

```json
[
    "int i...",
    "import json...",
    "func main...",
    ...
]
```

The command to run the code is:

```bash
python src/aiprogrammer.py --model_map model_map.json --input_path data/code_snippets.json --output_path data/aiprogrammer.json
```

## Data collection

Run the following scripts to generate data from various data sources:

```bash
# AIprogrammer
python src/data_collection.py --model_map model_map.json --data_type aiprogrammer --input_path data/aiprogrammer.json --output_path data/aiprogrammer_end.json

# Git Commit
python src/data_collection.py --model_map model_map.json --data_type commit --input_path data/commit.json --output_path data/commit_end.json --limit_one_block_prob 1.0

# Online Submit
python src/data_collection.py --model_map model_map.json --data_type submit --input_path data/submit_process.json --output_path data/submit_end.json
```

## Synthetic target area

Programmers often specify the parts requiring changes, typically in one of two ways: either by clicking with the cursor to indicate a general area or by selecting a specific text range with defined start and end points.

We synthesize the target modification area with a random algorithm:

```bash
python src/post_collection.py --input_path data/tmp.json --output_path data/tmp_area.json
```
