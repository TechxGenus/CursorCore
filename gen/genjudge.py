from .llmgen import LLMGen
from .template.genjudge_template import GENJUDGE_SYSTEM, GENJUDGE_RECORD_TYPE, GENJUDGE_PROMPT_INPUT, GENJUDGE_PROMPT_INPUT_RECORD, GENJUDGE_PROMPT_INPUT_CHANGE, GENJUDGE_PROMPT_OUTPUT, GENJUDGE_FEWSHOT

class GenJudgement(LLMGen):
    def __init__(self, backend="openai", model_map={}, max_try=5, num_proc=512, **kwargs) -> None:
        super().__init__(backend, model_map, max_try, num_proc, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def create_prompt(self, text_map):
        """
        Generates a prompt for the GENJUDGE model based on the provided text map.

        The prompt consists of a series of messages formatted for the model, including
        system instructions, few-shot examples, and the user input based on the given text map.

        Args:
            text_map (dict): A dictionary containing the 'record' and 'change' keys. 
                             'record' is a list of dictionaries, each representing a record with a 'type' key.
                             'change' is a list of changes to be included in the prompt.

        Returns:
            list: A list of dictionaries, each representing a message in the prompt. 
                  The messages alternate between 'user' and 'assistant' roles, with the final message being the user input.
        """
        out = [
            {"role": "system", "content": GENJUDGE_SYSTEM},
        ] + [
            {"role": "user", "content": GENJUDGE_PROMPT_INPUT.format_map(
                {
                    "record": GENJUDGE_PROMPT_INPUT_RECORD.format_map({"record": "\n".join(GENJUDGE_RECORD_TYPE[r["type"]].format_map(r) for r in shot["record"])}),
                    "change": "\n".join(GENJUDGE_PROMPT_INPUT_CHANGE.format_map({"num": i+1, "change": c}) for i, c in enumerate(shot["change"])),
                }
            )} if i % 2 == 0 else {"role": "assistant", "content": GENJUDGE_PROMPT_OUTPUT.format_map(shot)}
            for i, shot in enumerate(GENJUDGE_FEWSHOT)
        ] + [
            {
                "role": "user",
                "content": GENJUDGE_PROMPT_INPUT.format_map(
                    {
                        "record": GENJUDGE_PROMPT_INPUT_RECORD.format_map({"record": "\n".join(GENJUDGE_RECORD_TYPE[r["type"]].format_map(r) for r in text_map["record"])}),
                        "change": "\n".join(GENJUDGE_PROMPT_INPUT_CHANGE.format_map({"num": i+1, "change": c}) for i, c in enumerate(text_map["change"])),
                    }
                ),
            },
        ]
        return out

    def reject_response(self, text_map, response):
        """
        Determines whether a given response should be rejected based on specific criteria.

        Args:
            text_map (dict): A dictionary containing the text data, specifically with a key "change".
            response (str): The response string to be evaluated.

        Returns:
            bool: True if the response should be rejected, False otherwise.

        Criteria for rejection:
            - The response length is less than 20 characters.
            - The response contains the word "sorry" (case insensitive).
            - The number of segments in the response after splitting by "Analysis of change" does not match the number of changes in text_map["change"].
            - Any segment in the response does not contain "**Decision:**" or does not have "True" or "False" following "**Decision:**".
        """
        if len(response) < 20:
            return True
        if "sorry" in response.lower():
            return True
        each_judge = response.split("Analysis of change")[1:]
        if len(each_judge) != len(text_map["change"]):
            return True
        for judge in each_judge:
            if "**Decision:**" not in judge or ("True" not in judge.split("**Decision:**")[-1] and "False" not in judge.split("**Decision:**")[-1]):
                return True
        return False

    def post_process(self, text_map, response):
        """
        Processes the response text to extract decision outcomes.

        Args:
            text_map (dict): A mapping of text elements (not used in the function).
            response (str): The response string containing multiple judge analyses.

        Returns:
            list: A list of boolean values representing the decisions extracted from the response.
              Each boolean corresponds to a decision where True indicates a positive decision
              and False indicates a negative decision.

        Raises:
            AssertionError: If a decision is not found in any of the judge analyses.
        """
        each_judge = response.split("Analysis of change")[1:]
        out = []
        for judge in each_judge:
            if "True" in judge.split("**Decision:**")[-1]:
                out.append(True)
            elif "False" in judge.split("**Decision:**")[-1]:
                out.append(False)
            else:
                assert False, f"Decision not found in {judge}"
        return out
