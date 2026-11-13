from pydantic import Field

from .message import AnyMessage
from .meta import Meta


class Session(Meta):
    session_id: str
    messages: list[AnyMessage] = Field(default_factory=list)
