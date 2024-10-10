import random
import concurrent.futures

class LLMGen:
    def __init__(self, backend="openai", model_map={}, max_try=5, num_proc=512, **kwargs) -> None:
        self.model_name = list(model_map.keys())
        self.max_try = max_try
        if backend == "openai":
            from .openai import OpenAICompletion
            self.backend = OpenAICompletion(model_map)
        elif backend == "test":
            self.backend = None
        else:
            raise ValueError(
                f"backend {backend} is currently not supported"
            )
        self.kwargs = kwargs
        self.executor = concurrent.futures.ThreadPoolExecutor(num_proc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()

    def create_prompt(self, text_map, model_name):
        pass
    
    def reject_response(self, text_map, response):
        pass
    
    def post_process(self, text_map, response):
        pass

    def gen(
        self, text_maps, api_type="chat_completion"
    ):
        def process_text_map(text_map):
            """
            Processes a given text map by generating a response using a specified model.

            Args:
                text_map (dict): The input text map to be processed.

            Returns:
                dict: A dictionary containing the input text map and the processed output.
                  If the response is rejected after the maximum number of tries, the output will be None.

            Raises:
                ValueError: If the specified api_type is not supported.
            """
            success = False
            try_count = 0
            answer = ""
            while try_count < self.max_try and not success:
                if type(self.model_name) == list:
                    model_name = random.choice(self.model_name)
                else:
                    model_name = self.model_name
                input_text = self.create_prompt(text_map)
                if api_type == "chat_completion":
                    answer = self.backend.chat_completion(input_text, model_name, **self.kwargs)
                elif api_type == "completion":
                    answer = self.backend.completion(input_text, model_name, **self.kwargs)
                else:
                    raise ValueError(
                        f"api_type {api_type} is currently not supported"
                    )
                if self.reject_response(text_map, answer):
                    try_count += 1
                    continue
                success = True
            if self.reject_response(text_map, answer):
                return {"input": text_map, "output": None}
            else:
                return {"input": text_map, "output": self.post_process(text_map, answer)}

        gen = []
        results = self.executor.map(process_text_map, text_maps)
        for result in results:
            gen.append(result)
        return gen
