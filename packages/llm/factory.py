from packages.llm.providers.groq import GroqProvider
from packages.llm.types import LLMProvider

class LLMFactory:
    _providers = {
        LLMProvider.GROQ: GroqProvider,
    }

    @classmethod
    def create(cls, provider: LLMProvider):
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}")

        return cls._providers[provider]()