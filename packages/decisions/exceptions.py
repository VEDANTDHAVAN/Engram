from __future__ import annotations


class DecisionProviderError(Exception):
    """Base exception for decision providers."""


class LayaDecisionError(DecisionProviderError):
    """Raised when Laya cannot produce a valid decision."""