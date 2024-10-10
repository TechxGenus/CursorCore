from .llmgen import LLMGen
from .template.geninst_template import GENINST_SYSTEM, GENINST_PROMPT_INPUT, GENINST_PROMPT_OUTPUT, GENINST_FEWSHOT, GENINST_RECORD_TYPE


class GenInstruction(LLMGen):
    def __init__(self, backend="openai", model_map={}, max_try=5, num_proc=512, **kwargs) -> None:
        super().__init__(backend, model_map, max_try, num_proc, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def create_prompt(self, text_map):
        """
        Generates a prompt for the language model based on the provided text map and predefined few-shot examples.

        Args:
            text_map (dict): A dictionary containing the 'record' and 'change' keys. 
                             'record' is a list of dictionaries, each with a 'type' key.
                             'change' is a string describing the change.

        Returns:
            list: A list of dictionaries representing the prompt, where each dictionary has 'role' and 'content' keys.
        """
        out = [
            {"role": "system", "content": GENINST_SYSTEM},
        ] + [
            {"role": "user", "content": GENINST_PROMPT_INPUT.format_map(
                {
                    "record": "\n".join(GENINST_RECORD_TYPE[r["type"]].format_map(r) for r in shot["record"]),
                    "change": shot["change"],
                }
            )} if i % 2 == 0 else {"role": "assistant", "content": GENINST_PROMPT_OUTPUT.format_map(shot)}
            for i, shot in enumerate(GENINST_FEWSHOT)
        ] + [
            {
                "role": "user",
                "content": GENINST_PROMPT_INPUT.format_map(
                    {
                        "record": "\n".join(GENINST_RECORD_TYPE[r["type"]].format_map(r) for r in text_map["record"]),
                        "change": text_map["change"],
                    }
                ),
            },
        ]
        return out

    def reject_response(self, text_map, response):
        """
        Determines whether a given response should be rejected based on specific criteria.

        Args:
            text_map (dict): A dictionary containing text mappings (not used in the current implementation).
            response (str): The response string to be evaluated.

        Returns:
            bool: True if the response should be rejected, False otherwise.

        The response is rejected if:
        - It does not start with "**instruction:**" after stripping leading and trailing whitespace.
        - It contains the substring "\nNote:".
        - It contains the phrase "no change" (case insensitive).
        """
        if not response.strip().startswith("**instruction:**") or "\nNote:" in response or "no change" in response.lower():
            return True
        return False
    
    def post_process(self, text_map, response):
        """
        Post-processes the response by stripping leading and trailing whitespace and 
        splitting the text based on the delimiter "**instruction:**". It returns the 
        last segment of the split response.

        Args:
            text_map (dict): A dictionary mapping text segments (not used in this method).
            response (str): The response string to be post-processed.

        Returns:
            str: The processed response string.
        """
        return response.strip().split("**instruction:**")[-1].strip()
