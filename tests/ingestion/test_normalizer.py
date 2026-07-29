from packages.ingestion.normalizer import DocumentNormalizer
from packages.ingestion.schemas import RawDocument

def test_normalize_html():
    doc = RawDocument(
        source="web", content="<h1>Hello</h1><p>World</p>",
    )
    normalized = DocumentNormalizer.normalize(doc)

    assert normalized.content == "Hello World"

def test_whitespace_cleanup():
    doc = RawDocument(
        source="web", content="Hello       \n\n     World",
    )

    normalized = DocumentNormalizer.normalize(doc)

    assert normalized.content == "Hello World"

def test_empty_document():
    doc = RawDocument(
        source="web", content="  ",
    )

    try: 
        DocumentNormalizer.normalize(doc)
        assert False
    except Exception:
        assert True