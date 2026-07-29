from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

from .exceptions import NormalizationError
from .schemas import NormalizedDocument, RawDocument

class DocumentNormalizer:
    """Deterministic preprocessing before chunking."""
    @staticmethod
    def normalize(document: RawDocument) -> NormalizedDocument:
        if not document.content.strip():
            raise NormalizationError("Document content is empty.")

        text = document.content

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Strip HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

        # Remove control characters
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C"
            or ch in ("\n", "\t")
        )

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return NormalizedDocument(
            source=document.source, content=text,
            title=document.title, url=document.url,
            language=document.language, metadata=document.metadata,
        )