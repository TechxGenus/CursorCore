# APEval: Assist Programming Eval

This benchmark aims to assess how models use various types of information to assist programming. It is extended by the [HumanEval](https://github.com/openai/human-eval) benchmark. The benchmark is structured as a JSON file where each entry corresponds to a specific task. Below is a detailed description of the fields present in the benchmark.

## File Structure

The dataset is in JSON format with each entry structured as follows:

- **task_id**: A unique identifier for each task, corresponding to the original HumanEval benchmark.
- **history**: An array of historical code snippets and their related metadata. Each snippet in the history represents a different version of code.
  - **type**: The type of content, typically `code`.
  - **lang**: The programming language used in the code, typically `python`.
  - **code**: The historical code snippets from different moments.
- **current**: The current code and its related metadata:
  - **type**: The type of content, typically `code`.
  - **lang**: The programming language used in the code, typically `python`.
  - **code**: The current version of the code for the task.
- **user**: An instruction or reflextion provided by the user regarding the task.
- **area**: Extra metadata, indicates the location of the cursor or the selected code area.

## Example Case

```json
{
  "task_id": "HumanEval/0",
  "history": [
    {
      "type": "code",
      "lang": "python",
      "code": "def has_close_elements(n, t):\n    for i in range(prm)"
    },
    ...
  ],
  "current": {
    "type": "code",
    "lang": "python",
    "code": "def has_close_elements(n, t):\n    for i in range(len(n - 1)):\n        for j in range(i + 1, len(n)):\n            if n[i] - n[j] < t or n[j] - n[i] < t:"
  },
  "user": "",
  "area": 151
}
```
