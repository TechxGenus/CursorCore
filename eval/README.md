# Evaluation

This folder contains scripts for evaluating models.

## Inference service for LLMs

Similar to data collection, we uniformly use the OpenAI interface to generate.

**Note**: We leverages extra parameters specific to `vllm`'s OpenAI-compatible server for handling custom chat templates and special tokens for our models. Other OpenAI-compatible inference services may not be directly applicable.

Example script to deploy `CursorCore-Yi-1.5B` using `vllm`:

```bash
python -m vllm.entrypoints.openai.api_server --port 10086 --model TechxGenus/CursorCore-Yi-1.5B
```

We define the model inference service parameters in `model_map.json`. An example configuration is as follows:

```json
{
    "TechxGenus/CursorCore-Yi-1.5B": {
        "base": "http://127.0.0.1:10086/v1",
        "api": "sk-xxx"
    }
}
```

## Run APEval evaluation

Run the following program to generate predicted code:

```bash
# WF Format (Default)
python eval/eval_apeval.py --model_map model_map.json --input_path benchmark/apeval.json --output_path eval/generations.jsonl --temperature 0.0 --use_wf

# LC Format
python eval/eval_apeval.py --model_map model_map.json --input_path benchmark/apeval.json --output_path eval/generations.jsonl --temperature 0.0 --use_lc

# SR Format
python eval/eval_apeval.py --model_map model_map.json --input_path benchmark/apeval.json --output_path eval/generations.jsonl --temperature 0.0 --use_sr

# Instruct Models
python eval/eval_apeval.py --model_map model_map.json --input_path benchmark/apeval.json --output_path eval/generations.jsonl --temperature 0.0 --use_instruct

# Base Models
python eval/eval_apeval.py --model_map model_map.json --input_path benchmark/apeval.json --output_path eval/generations.jsonl --temperature 0.0 --use_base
```

Run the following script to execute programs:

```bash
evalplus.evaluate --dataset humaneval --samples eval/generations.jsonl
```

Run the following script to get evaluation results for each type:

```bash
python eval/extract_results.py --dataset_path benchmark/apeval.json --result_path eval/generations_eval_results.json
```

## Run HumanEval/MBPP evaluation

Run the following program to generate predicted code:

```bash
# Tab
python eval/eval_humaneval.py --model_map model_map.json --input_path evalplus/humanevalplus --output_path eval/generations.jsonl --temperature 0.0 --use_tab
python eval/eval_mbpp.py --model_map model_map.json --input_path evalplus/mbppplus --output_path eval/generations.jsonl --temperature 0.0 --use_tab

# Inline
python eval/eval_humaneval.py --model_map model_map.json --input_path evalplus/humanevalplus --output_path eval/generations.jsonl --temperature 0.0 --use_inline
python eval/eval_mbpp.py --model_map model_map.json --input_path evalplus/mbppplus --output_path eval/generations.jsonl --temperature 0.0 --use_inline

# Chat
python eval/eval_humaneval.py --model_map model_map.json --input_path evalplus/humanevalplus --output_path eval/generations.jsonl --temperature 0.0 --use_chat
python eval/eval_mbpp.py --model_map model_map.json --input_path evalplus/mbppplus --output_path eval/generations.jsonl --temperature 0.0 --use_chat
```

Run the following script to execute programs:

```bash
evalplus.evaluate --dataset humaneval --samples eval/generations.jsonl
evalplus.evaluate --dataset mbpp --samples eval/generations.jsonl
```
