from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field


class GetCurrentTimeInput(BaseModel):
    tz_offset_hours: int = Field(default=0, description="UTC offset in hours")


class GetCurrentTimeOutput(BaseModel):
    iso_time: str = Field(description="Current date/time as ISO 8601")


def get_current_time(tz_offset_hours: int = 0) -> str:
    """Return the current date and time as ISO 8601 for a given UTC offset in hours."""
    return datetime.now(timezone(timedelta(hours=tz_offset_hours))).isoformat()
