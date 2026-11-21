from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

from pydantic import BaseModel, Field

from .definition import ToolDefinition

_ALL_ZONES = available_timezones()

_CITY_TO_ZONE: dict[str, str] = {}
for _tz in _ALL_ZONES:
    parts = _tz.split("/")
    if len(parts) >= 2:
        city = parts[-1].lower().replace("_", " ")
        _CITY_TO_ZONE[city] = _tz


class GetTimezoneOffsetInput(BaseModel):
    location: str = Field(description="City name or IANA timezone (e.g. 'London', 'Asia/Bahrain', 'New York')")


def get_timezone_offset(location: str) -> str:
    """Look up the current UTC offset for a city or IANA timezone."""
    query = location.strip()

    if query in _ALL_ZONES:
        zone = ZoneInfo(query)
        offset = datetime.now(zone).utcoffset()
        hours = offset.total_seconds() / 3600
        return f"{query} UTC{hours:+.1f}"

    key = query.lower().replace("_", " ")
    if key in _CITY_TO_ZONE:
        tz_name = _CITY_TO_ZONE[key]
        zone = ZoneInfo(tz_name)
        offset = datetime.now(zone).utcoffset()
        hours = offset.total_seconds() / 3600
        return f"{tz_name} UTC{hours:+.1f}"

    matches = [tz for city, tz in _CITY_TO_ZONE.items() if key in city]
    if len(matches) == 1:
        tz_name = matches[0]
        zone = ZoneInfo(tz_name)
        offset = datetime.now(zone).utcoffset()
        hours = offset.total_seconds() / 3600
        return f"{tz_name} UTC{hours:+.1f}"
    if matches:
        return f"Ambiguous: {', '.join(sorted(matches[:10]))}. Be more specific or use an IANA timezone."

    return f"Unknown location '{location}'. Use an IANA timezone like 'Europe/London' or a city name."


def _extractor(output: str):
    return [(output, "exact")]


TOOL = ToolDefinition(
    name="get_timezone_offset",
    description="Look up the current UTC offset for a city or IANA timezone. Call this BEFORE get_current_time to find the correct offset.",
    inputSchema=GetTimezoneOffsetInput.model_json_schema(),
    kind="fetch",
    fn=get_timezone_offset,
    extractor=_extractor,
)
