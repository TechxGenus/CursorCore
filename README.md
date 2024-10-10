# CursorCore: Assist Programming through Aligning Anything

<p align="center">
<a href="http://arxiv.org/abs/2410.07002">[📄arXiv]</a> |
<a href="https://hf.co/papers/2410.07002">[🤗HF Paper]</a> |
<a href="https://huggingface.co/collections/TechxGenus/cursorcore-series-6706618c38598468866b60e2">[🤖Models]</a> |
<a href="https://github.com/TechxGenus/CursorCore">[🛠️Code]</a> |
<a href="https://github.com/TechxGenus/CursorWeb">[<img src="https://github.com/TechxGenus/CursorCore/blob/main/pictures/cursorcore.png" width="12.5px">Web]</a> |
<a href="https://discord.gg/Z5Tev8fV">[<img src="https://github.com/TechxGenus/CursorCore/blob/main/pictures/discord.png" width="15x">Discord]</a>
</p>

<hr>

- [CursorCore: Assist Programming through Aligning Anything](#cursorcore-assist-programming-through-aligning-anything)
  - [Introduction](#introduction)
  - [Structure](#structure)
  - [Models](#models)
  - [Usage](#usage)
    - [1) Normal chat](#1-normal-chat)
    - [2) Assistant-Conversation](#2-assistant-conversation)
    - [3) Web Demo](#3-web-demo)
  - [Future Work](#future-work)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)
  - [Contribution](#contribution)

<hr>

## Introduction

CursorCore is a series of open-source models designed for AI-assisted programming. It aims to support features such as automated editing and inline chat, replicating the core abilities of closed-source AI-assisted programming tools like Cursor. This is achieved by aligning data generated through Programming-Instruct. Please read [our paper](http://arxiv.org/abs/2410.07002) to learn more.

<p align="center">
<img width="100%" alt="conversation" src="https://github.com/TechxGenus/CursorCore/blob/main/pictures/conversation.png">
</p>

![CursorWeb](https://github.com/TechxGenus/CursorCore/blob/main/pictures/CursorWeb.gif)

## Structure

- `./benchmark` contains the APEval benchmark
- `./data` contains code to preprocess datasets
- `./eval` contains code to evaluate models
- `./gen` contains code to prompt LLMs for generation
- `./generic` common functions, tools and special tokens
- `./src` contains code about Programming-Instruct
- `./train` contains code for training CursorCore

Please ensure all dependencies are installed using the following command:

```bash
pip install -r requirements.txt
```

We also use [flash-attention](https://github.com/Dao-AILab/flash-attention) for efficient training and [flashinfer](https://github.com/flashinfer-ai/flashinfer) to accelerate inference. See the documents for them to learn how to install.

## Models

Our models have been open-sourced on Hugging Face. You can access our models here: [CursorCore-Series](https://huggingface.co/collections/TechxGenus/cursorcore-series-6706618c38598468866b60e2"). We also provide pre-quantized weights for GPTQ and AWQ here: [CursorCore-Quantization](https://huggingface.co/collections/TechxGenus/cursorcore-quantization-67066431f29f252494ee8cf3)

We use the manually written benchmark APEval to assess the model's ability to assist programming. We also utilize [EvalPlus](https://github.com/evalplus/evalplus), [CanItEdit](https://github.com/nuprl/CanItEdit) and [OctoPack](https://github.com/bigcode-project/octopack) to evaluate the model's performance in Python program generation, instructional code editing, and automated program repair. Since we use a custom conversation template, its generation method differs significantly from both instruct models and base models. Please refer to [our paper](http://arxiv.org/abs/2410.07002) for more details.

Evaluation results on APEval:

<img src="https://github.com/TechxGenus/CursorCore/blob/main/pictures/APEval.png" alt="APEval" width="75%"/>

Evaluation results on EvalPlus, CanItEdit and OctoPack:

<img src="https://github.com/TechxGenus/CursorCore/blob/main/pictures/EvalPlus_CanItEdit_OctoPack.png" alt="EvalPlus_CanItEdit_OctoPack" width="75%">

## Usage

Here are some examples of how to use our model:

### 1) Normal chat

Script:

````python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/CursorCore-Yi-9B")
model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/CursorCore-Yi-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Hi!"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
````

Output:

````txt
<|im_start|>system
You are a helpful programming assistant.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
Hello! I'm an AI language model and I can help you with any programming questions you might have. What specific problem or task are you trying to solve?<|im_end|>
````

### 2) Assistant-Conversation

In our work, we introduce a new framework of AI-assisted programming task. It is designed for aligning anything during programming process, used for the implementation of features like Tab and Inline Chat.

Script 1:

````python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.utils import prepare_input_for_wf

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/CursorCore-Yi-9B")
model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/CursorCore-Yi-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
sample = {
    "history": [
        {
            "type": "code",
            "lang": "python",
            "code": """def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
        }
    ],
    "current": {
        "type": "code",
        "lang": "python",
        "code": """def quick_sort(array):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
    },
    "user": ""
}

prompt = tokenizer.apply_chat_template(
    prepare_input_for_wf(sample),
    tokenize=False,
    chat_template="assistant-conversation",
    add_generation_prompt=True
)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0]))
````

Output 1:

````txt
<|im_start|>system
You are a helpful programming assistant.<|im_end|>
<|im_start|>history
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|im_end|>
<|im_start|>current
```python
def quick_sort(array):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|im_end|>
<|im_start|>assistant
<|next_start|>```python
def quick_sort(array):
    if len(array) <= 1:
        return array
    pivot = array[len(array) // 2]
    left = [x for x in array if x < pivot]
    middle = [x for x in array if x == pivot]
    right = [x for x in array if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|next_end|>
The user has provided a revised code change that corrects the parameter name from `arr` to `array` in the `quick_sort` function. This change ensures consistency in the function definition and avoids potential confusion or errors.

To implement this, we will:
1. Update the parameter name in the function definition from `arr` to `array`.
2. Ensure that all references to `arr` within the function are updated to `array`.

This will make the function definition and internal references consistent, improving code readability and maintainability.<|im_end|>
````

Script 2:

````python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.utils import prepare_input_for_wf

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/CursorCore-Yi-9B")
model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/CursorCore-Yi-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
sample = {
    "history": [],
    "current": {
        "type": "code",
        "lang": "python",
        "code": """def quick_sort(array):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
    },
    "user": "Add Docstring."
}

prompt = tokenizer.apply_chat_template(
    prepare_input_for_wf(sample),
    tokenize=False,
    chat_template="assistant-conversation",
    add_generation_prompt=True
)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0]))
````

Output 2:

````txt
<|im_start|>system
You are a helpful programming assistant.<|im_end|>
<|im_start|>current
```python
def quick_sort(array):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|im_end|>
<|im_start|>user
Add Docstring.<|im_end|>
<|im_start|>assistant
<|next_start|>```python
def quick_sort(array):
    """
    This is an implementation of the quick sort algorithm.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|next_end|><|im_end|>
````

For models in Locate-and-Change (LC) and Search-and-Replace (SR) formats, the output examples are as follows:

Script for LC:

````python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.utils import prepare_input_for_lc

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/CursorCore-Yi-1.5B-LC")
model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/CursorCore-Yi-1.5B-LC",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
sample = {
    "history": [],
    "current": {
        "type": "code",
        "lang": "python",
        "code": """def quick_sort(array):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
    },
    "user": "Add Docstring."
}

prompt = tokenizer.apply_chat_template(
    prepare_input_for_lc(sample),
    tokenize=False,
    chat_template="assistant-conversation",
    add_generation_prompt=True
)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0]))
````

Output for LC:

````txt
<|im_start|>system
You are a helpful programming assistant.<|im_end|>
<|im_start|>current
```python
1 def quick_sort(array):
2     if len(arr) <= 1:
3         return arr
4     pivot = arr[len(arr) // 2]
5     left = [x for x in arr if x < pivot]
6     middle = [x for x in arr if x == pivot]
7     right = [x for x in arr if x > pivot]
8     return quick_sort(left) + middle + quick_sort(right)
```<|im_end|>
<|im_start|>user
Add Docstring.<|im_end|>
<|im_start|>assistant
<|next_start|>1,1
```
    '''This function will sort an array using quick sort algorithm'''
```<|next_end|>
To enhance the readability and maintainability of the code, we should add a docstring to the `quick_sort` function. A docstring provides a clear description of what the function does, which is particularly useful for other developers who might use or modify this code in the future.

The docstring will be added immediately after the function definition, explaining that the function uses the quick sort algorithm to sort an array. This will make the code more self-explanatory and easier to understand.

Here's the plan:
1. Add a docstring at the beginning of the `quick_sort` function.
2. Ensure the docstring is clear and concise, describing the purpose of the function.

This modification will improve the code's documentation without altering its functionality.<|im_end|>
````

Script for SR:

````python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.utils import prepare_input_for_sr

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/CursorCore-Yi-1.5B-SR")
model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/CursorCore-Yi-1.5B-SR",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
sample = {
    "history": [],
    "current": {
        "type": "code",
        "lang": "python",
        "code": """def quick_sort(array):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
    },
    "user": "Add Docstring."
}

prompt = tokenizer.apply_chat_template(
    prepare_input_for_sr(sample),
    tokenize=False,
    chat_template="assistant-conversation",
    add_generation_prompt=True
)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0]))
````

Output for SR:

````txt
<|im_start|>system
You are a helpful programming assistant.<|im_end|>
<|im_start|>current
```python
def quick_sort(array):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```<|im_end|>
<|im_start|>user
Add Docstring.<|im_end|>
<|im_start|>assistant
<|next_start|>```python
def quick_sort(array):
<|search_and_replace|>
def quick_sort(array):
    """
    This function implements quick sort algorithm
    """
```<|next_end|><|im_end|>
````

### 3) Web Demo

We create a web demo for CursorCore. Please visit [CursorWeb](https://github.com/TechxGenus/CursorWeb) for more details.

## Future Work

CursorCore is still in a very early stage, and lots of work is needed to achieve a better user experience. For example:

- Repository-level editing support
- Better and faster editing formats
- Better user interface and presentation
- ...

## Citation

```bibtex
@article{jiang2024cursorcore,
  title   = {CursorCore: Assist Programming through Aligning Anything},
  author  = {Hao Jiang and Qi Liu and Rui Li and Shengyu Ye and Shijin Wang},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2410.07002}
}
```

## Acknowledgements

The open-source community has been of great help to us, and we reference numerous projects and applications. They include but are not limited to:

[Deepseek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder), [Yi-Coder](https://github.com/01-ai/Yi-Coder), [Qwen-Coder](https://github.com/QwenLM/Qwen2.5-Coder), [Self-Instruct](https://github.com/yizhongw/self-instruct), [Evol-Instruct](https://github.com/theblackcat102/evol-dataset), [OSS-Instruct](https://github.com/ise-uiuc/magicoder), [EvalPlus](https://github.com/evalplus/evalplus), [CanItEdit](https://github.com/nuprl/CanItEdit), [OctoPack](https://github.com/bigcode-project/octopack), [Aider](https://github.com/Aider-AI/aider), [Continue](https://github.com/continuedev/continue), [Cursor](https://github.com/getcursor/cursor), ...

## Contribution

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
