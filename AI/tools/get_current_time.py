from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from .definition import ToolDefinition


class GetCurrentTimeInput(BaseModel):
    tz_offset_hours: int = Field(default=0, description="UTC offset in hours")


def get_current_time(tz_offset_hours: int = 0) -> str:
    """Return the current date and time as ISO 8601 for a given UTC offset."""
    return datetime.now(timezone(timedelta(hours=tz_offset_hours))).isoformat()


def _extractor(output: str):
    return [(output, "temporal")]


TOOL = ToolDefinition(
    name="get_current_time",
    description="Return the current date and time as ISO 8601 for a given UTC offset. Use get_timezone_offset first to find the offset.",
    inputSchema=GetCurrentTimeInput.model_json_schema(),
    fn=get_current_time,
    extractor=_extractor,
    depends_on=["get_timezone_offset"],
)
