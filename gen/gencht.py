from .llmgen import LLMGen
from .template.gencht_template import GENCHAT_SYSTEM, GENCHAT_PROMPT_INPUT, GENCHAT_PROMPT_OUTPUT, GENCHAT_FEWSHOT, GENCHAT_RECORD_TYPE


class GenChat(LLMGen):
    def __init__(self, backend="openai", model_map={}, max_try=5, num_proc=512, **kwargs) -> None:
        super().__init__(backend, model_map, max_try, num_proc, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def create_prompt(self, text_map):
        """
        Generates a prompt for a conversational AI model based on provided text mappings.

        Args:
            text_map (dict): A dictionary containing the following keys:
                - "record" (list): A list of dictionaries, each representing a record with a "type" key.
                - "change" (str): A string describing the change.

        Returns:
            list: A list of dictionaries, each representing a message in the conversation. Each dictionary contains:
                - "role" (str): The role of the message sender, either "system", "user", or "assistant".
                - "content" (str): The content of the message.
        """
        out = [
            {"role": "system", "content": GENCHAT_SYSTEM},
        ] + [
            {"role": "user", "content": GENCHAT_PROMPT_INPUT.format_map(
                {
                    "record": "\n".join(GENCHAT_RECORD_TYPE[r["type"]].format_map(r) for r in shot["record"]),
                    "change": shot["change"],
                }
            )} if i % 2 == 0 else {"role": "assistant", "content": GENCHAT_PROMPT_OUTPUT.format_map(shot)}
            for i, shot in enumerate(GENCHAT_FEWSHOT)
        ] + [
            {
                "role": "user",
                "content": GENCHAT_PROMPT_INPUT.format_map(
                    {
                        "record": "\n".join(GENCHAT_RECORD_TYPE[r["type"]].format_map(r) for r in text_map["record"]),
                        "change": text_map["change"],
                    }
                ),
            },
        ]
        return out

    def reject_response(self, text_map, response):
        """
        Determines whether a given response should be rejected based on its format.

        Args:
            text_map (dict): A dictionary containing text mappings (not used in the current implementation).
            response (str): The response string to be evaluated.

        Returns:
            bool: True if the response should be rejected (i.e., it does not start with "**chat:**"), False otherwise.
        """
        if not response.strip().startswith("**chat:**"):
            return True
        return False
    
    def post_process(self, text_map, response):
        """
        Post-processes the response by stripping whitespace and extracting the relevant part.

        Args:
            text_map (dict): A dictionary containing text mappings (not used in this function).
            response (str): The response string to be processed.

        Returns:
            str: The processed response, which is the part after the last occurrence of "**chat:**".
        """
        return response.strip().split("**chat:**")[-1].strip()
