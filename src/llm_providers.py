from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str


class LocalHFGenerator:
    """
    CPU-friendly open-source generator for RAG.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 450):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> LLMResponse:
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            truncation=True,
        )
        return LLMResponse(text=out[0]["generated_text"].strip())
