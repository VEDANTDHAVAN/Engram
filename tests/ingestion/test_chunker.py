from packages.ingestion.chunker import DocumentChunker
from packages.ingestion.schemas import NormalizedDocument

def test_single_chunk():
    doc = NormalizedDocument(
        source="web",
        content="Hello world",
    )
    chunker = DocumentChunker(chunk_size=100)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].token_start == 0
    assert chunks[0].token_end <= 128

    for i in range(1, len(chunks)):
        assert chunks[i].token_start < chunks[i].token_end
        assert chunks[i].token_start == (
            chunks[i - 1].token_end - chunker.overlap
        )

def test_multiple_chunks():
    text = " ".join(["Engram"] * 1000)
    doc = NormalizedDocument(
        source="web",
        content=text,
    )
    chunker = DocumentChunker(
        chunk_size=128,
        overlap=16,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= 128

def test_empty_document():
    doc = NormalizedDocument(
        source="web",
        content="",
    )
    chunker = DocumentChunker()

    try:
        chunker.chunk(doc)
        assert False
    except Exception:
        assert True

def test_overlap():
    text = " ".join(["memory"] * 300)

    doc = NormalizedDocument(
        source="web",
        content=text,
    )

    chunker = DocumentChunker(
        chunk_size=64,
        overlap=16,
    )

    chunks = chunker.chunk(doc)

    assert len(chunks) > 1

    for i in range(1, len(chunks)):
        assert (
            chunks[i].token_start
            == chunks[i - 1].token_end - 16
        )