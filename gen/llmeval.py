from .llmgen import LLMGen


class GenEvaluation(LLMGen):
    def __init__(self, backend="openai", model_map={}, max_try=1, num_proc=512, **kwargs) -> None:
        super().__init__(backend, model_map, max_try, num_proc, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def create_prompt(self, text_map):
        return text_map["conversation"]

    def reject_response(self, text_map, response):
        return False
    
    def post_process(self, text_map, response):
        return response
