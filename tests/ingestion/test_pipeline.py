from packages.ingestion import IngestionPipeline, RawDocument

def test_pipeline():
    doc = RawDocument(
        source="web", content="""
      <h1>Engram</h1>

      This is a Production Memory System.
    """,
    )

    pipeline = IngestionPipeline()
    result = pipeline.run(doc)

    assert result.document.content.startswith("Engram")
    assert result.total_chunks >= 1
    assert len(result.chunks) == result.total_chunks

def test_pipeline_preserves_metadata():
    doc = RawDocument(
        source="web", content="Hello",
        metadata={"project": "engram"},
    )

    result = IngestionPipeline().run(doc)

    assert result.document.metadata["project"] == "engram"