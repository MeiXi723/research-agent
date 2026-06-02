from openai import OpenAI
from fastembed import TextEmbedding

_llm_service = None


class LLMService:
    def __init__(self, api_key: str, base_url: str):
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    def embed(self, text: str) -> list[float]:
        return list(self.embed_model.embed([text]))[0].tolist()

    def chat(self, messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.7) -> str:
        resp = self.chat_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content
