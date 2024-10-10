from openai import OpenAI

#TODO: Support batch api calls
class OpenAICompletion:
    def __init__(
        self,
        model_map,
    ) -> None:
        self.client = {
            model: OpenAI(
                base_url=model_map[model]["base"],
                api_key=model_map[model]["api"],
            )
            for model in model_map
        }

    def chat_completion(
        self,
        prompt,
        model_name="deepseek-chat",
        temperature=0.2,
        max_tokens=3072,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|eot_id|>", "<|im_end|>", "</s>", "<|EOT|>", "<|endoftext|>", "<|eos|>"], # default stop tokens
        timeout=200,
        extra_body={},
    ):
        response = ""
        client = self.client[model_name]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                timeout=timeout,
                extra_body=extra_body,
            )
        except Exception as e:
            print("Exception", e)
        if response != "":
            if response.choices[0].finish_reason == "length":
                return ""
            return response.choices[0].message.content
        return ""

    def completion(
        self,
        prompt,
        model_name="deepseek-chat",
        temperature=0.2,
        max_tokens=3072,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["</s>", "<|endoftext|>", "<|eos_token|>"], # default stop tokens
        timeout=200,
        extra_body={},
    ):
        response = ""
        client = self.client[model_name]
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                timeout=timeout,
                extra_body=extra_body,
            )
        except Exception as e:
            print("Exception", e)
        if response != "":
            return response.choices[0].text
        return ""
