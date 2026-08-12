from packages.providers.vector import TurboVecProvider, VectorFactory
from packages.core.config import settings

def test_factory_creates_turbovec_provider():
    provider = VectorFactory.create(provider="turbovec")

    assert isinstance(provider, TurboVecProvider)

def test_factory_uses_case_insensitive_provider_name():
    provider = VectorFactory.create(provider="TURBOVEC")

    assert isinstance(provider, TurboVecProvider)

def test_factory_rejects_unknown_provider():
    try:
        VectorFactory.create(provider="unknown")
    except ValueError as exc:
        assert "Unsupported vector provider" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for unsupported vector provider"
        )

def test_turbovec_configuration():
    assert settings.VECTOR_PROVIDER == "turbovec"

    assert (
        settings.TURBOVEC_INDEX_PATH == "data/turbovec/semantic_memory.tvim"
    )

    assert settings.TURBOVEC_BIT_WIDTH == 4