from packages.ingestion.schemas import RawDocument

def test_raw_document():
    doc = RawDocument(
        source="web",
        content="Hello Engram",
    )

    assert doc.source == "web"
    assert doc.content == "Hello Engram"