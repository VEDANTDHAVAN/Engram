class GraphProviderError(Exception):
    """Base graph provider error."""


class GraphTraversalError(GraphProviderError):
    """Graph traversal failed."""