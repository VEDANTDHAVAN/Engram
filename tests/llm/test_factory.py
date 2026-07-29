from packages.llm.factory import LLMFactory
from packages.llm.types import LLMProvider
from packages.llm.providers.groq import GroqProvider

def test_factory_returns_groq():
    provider = LLMFactory.create(LLMProvider.GROQ)

    assert isinstance(provider, GroqProvider)