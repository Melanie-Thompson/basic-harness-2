"""Archived cluster model for TDA-based context management."""

from AI.models.message import AnyMessage
from AI.models.meta import Meta


class ArchivedCluster(Meta):
    centroid: list[float]
    messages: list[AnyMessage]
    token_count: int
