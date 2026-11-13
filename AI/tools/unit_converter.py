from pydantic import BaseModel, Field

from .definition import ToolDefinition

CONVERSIONS: dict[tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("celsius", "fahrenheit"): None,
    ("fahrenheit", "celsius"): None,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
    ("liters", "gallons"): 0.264172,
    ("gallons", "liters"): 3.78541,
    ("cm", "inches"): 0.393701,
    ("inches", "cm"): 2.54,
    ("grams", "ounces"): 0.035274,
    ("ounces", "grams"): 28.3495,
}


class UnitConverterInput(BaseModel):
    value: float = Field(description="Numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. 'km', 'celsius', 'kg')")
    to_unit: str = Field(description="Target unit (e.g. 'miles', 'fahrenheit', 'lbs')")


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common units. Supports km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons, cm/inches, grams/ounces."""
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    if f == "celsius" and t == "fahrenheit":
        return f"{value * 9/5 + 32:.2f} fahrenheit"
    if f == "fahrenheit" and t == "celsius":
        return f"{(value - 32) * 5/9:.2f} celsius"

    factor = CONVERSIONS.get((f, t))
    if factor is None:
        return f"error: unsupported conversion from '{from_unit}' to '{to_unit}'"

    return f"{value * factor:.4f} {to_unit}"


def _extractor(output: str):
    if output.startswith("error:"):
        return [(output, "exact")]
    return [(output, "numeric")]


TOOL = ToolDefinition(
    name="unit_converter",
    description="Convert a value between common units. Supports km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons, cm/inches, grams/ounces.",
    inputSchema=UnitConverterInput.model_json_schema(),
    fn=unit_converter,
    extractor=_extractor,
)
