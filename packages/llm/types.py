from enum import Enum

class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"