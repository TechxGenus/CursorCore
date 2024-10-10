import random
from .llmgen import LLMGen
from .template.aiprogrammer_template import NOVICE_AIPROGRAMMER_SYSTEM, ORDINARY_AIPROGRAMMER_SYSTEM, EXPERT_AIPROGRAMMER_SYSTEM, AIPROGRAMMER_PROMPT_INPUT, AIPROGRAMMER_PROMPT_OUTPUT, NOVICE_AIPROGRAMMER_FEWSHOT, ORDINARY_AIPROGRAMMER_FEWSHOT, EXPERT_AIPROGRAMMER_FEWSHOT
from .utils import extract_code_blocks

class AIProgrammer(LLMGen):
    def __init__(self, backend="openai", model_map={}, max_try=5, **kwargs) -> None:
        super().__init__(backend, model_map, max_try, **kwargs)

    def create_prompt(self, text_map):
        """
        Generates a prompt for an AI programmer based on a random skill level.

        The function randomly selects a skill level from "NOVICE", "ORDINARY", or "EXPERT".
        Based on the selected skill level, it sets the appropriate system message and few-shot examples.
        It then constructs a list of messages that includes the system message, alternating user and assistant
        messages from the few-shot examples, and a final user message based on the provided text_map.

        Args:
            text_map (dict): A dictionary containing the input text to be formatted into the final user message.

        Returns:
            list: A list of dictionaries representing the prompt messages for the AI programmer.
        """
        dice = random.choice(["NOVICE", "ORDINARY", "EXPERT"])
        if dice == "NOVICE":
            AIPROGRAMMER_SYSTEM = NOVICE_AIPROGRAMMER_SYSTEM
            AIPROGRAMMER_FEWSHOT = NOVICE_AIPROGRAMMER_FEWSHOT
        elif dice == "ORDINARY":
            AIPROGRAMMER_SYSTEM = ORDINARY_AIPROGRAMMER_SYSTEM
            AIPROGRAMMER_FEWSHOT = ORDINARY_AIPROGRAMMER_FEWSHOT
        else:
            AIPROGRAMMER_SYSTEM = EXPERT_AIPROGRAMMER_SYSTEM
            AIPROGRAMMER_FEWSHOT = EXPERT_AIPROGRAMMER_FEWSHOT
        out = [
            {"role": "system", "content": AIPROGRAMMER_SYSTEM},
        ] + [
            {
            "role": "user",
            "content": AIPROGRAMMER_PROMPT_INPUT.format_map(shot)
            } if i % 2 == 0 else {
            "role": "assistant",
            "content": AIPROGRAMMER_PROMPT_OUTPUT.format_map(shot)
            } for i, shot in enumerate(AIPROGRAMMER_FEWSHOT)
        ] + [
            {"role": "user", "content": AIPROGRAMMER_PROMPT_INPUT.format_map(text_map)}
        ]
        return out
        
    def reject_response(self, text_map, response):
        """
        Determines whether a given response should be rejected based on various criteria.

        Args:
            text_map (dict): A dictionary containing the original input text with a key "content".
            response (str): The response text to be evaluated.

        Returns:
            bool: True if the response should be rejected, False otherwise.

        The function checks the following conditions to decide if the response should be rejected:
        - The response length is less than 20 characters.
        - The response contains the word "sorry".
        - The response does not contain any code blocks or the last code block is not the same as the input code.
        - The response contains repeated code blocks.
        - The response contains a summary of the process and repeats the last block.
        - The response contains segments of code where the sum of the lengths of all but the last block is less than or equal to the length of the last block plus 2 lines.
        - The response contains any consecutive identical code blocks.
        """
        if len(response) < 20:
            return True
        if "sorry" in response.lower():
            return True
        try:
            blocks = extract_code_blocks(response)
            # if the last code block is not the same as the input code
            if len(blocks) == 0 or blocks[-1].strip() != text_map["content"].strip():
                return True
            # sometimes llm will summarize the process and repeat the last block
            if len(blocks) >= 2 and blocks[-2] == blocks[-1]:
                blocks = blocks[:-1]
            block_lengths = [len(block.split("\n")) for block in blocks]
            # filter each step is just a segment of the whole code
            if len(block_lengths) >= 4 and sum(block_lengths[:-1]) <= block_lengths[-1] + 2:
                return True
            # filter if any block is the same during the process
            elif any(blocks[i] == blocks[i + 1] for i in range(len(blocks) - 1)):
                return True
        except:
            return True
        return False
    
    def post_process(self, text_map, response):
        """
        Post-processes the response by extracting code blocks and removing any repeated blocks.

        Args:
            text_map (dict): A mapping of text elements.
            response (str): The response string containing code blocks.

        Returns:
            list: A list of extracted code blocks with any repeated blocks removed.
        """
        blocks = extract_code_blocks(response)
        # sometimes llm will summarize the process and repeat the last block
        if len(blocks) >= 2 and blocks[-2] == blocks[-1]:
            blocks = blocks[:-1]
        return blocks
