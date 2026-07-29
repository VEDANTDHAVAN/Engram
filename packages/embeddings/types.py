from enum import Enum

class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    VOYAGE = "voyage"
    JINA = "jina"
    COHERE = "cohere"