import random

from pydantic import BaseModel, Field

from .definition import ToolDefinition


class DiceRollerInput(BaseModel):
    notation: str = Field(description="Dice notation like '2d6', '1d20', '3d8+5'")


def dice_roller(notation: str) -> str:
    """Roll dice using standard notation (e.g. '2d6', '1d20+3', '4d8-2'). Returns individual rolls and total."""
    notation = notation.strip().lower()

    modifier = 0
    base = notation
    for op in ("+", "-"):
        if op in notation:
            parts = notation.split(op, 1)
            base = parts[0]
            try:
                mod_val = int(parts[1])
                modifier = mod_val if op == "+" else -mod_val
            except ValueError:
                return f"error: invalid modifier in '{notation}'"
            break

    if "d" not in base:
        return f"error: invalid dice notation '{notation}', expected format like '2d6'"

    try:
        count_str, sides_str = base.split("d", 1)
        count = int(count_str) if count_str else 1
        sides = int(sides_str)
    except ValueError:
        return f"error: invalid dice notation '{notation}'"

    if count < 1 or count > 100:
        return "error: roll between 1 and 100 dice"
    if sides < 2 or sides > 100:
        return "error: dice must have between 2 and 100 sides"

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + modifier

    rolls_str = ", ".join(str(r) for r in rolls)
    mod_str = f" {'+' if modifier >= 0 else ''}{modifier}" if modifier else ""
    return f"[{rolls_str}]{mod_str} = {total}"


def _extractor(output: str):
    if output.startswith("error:"):
        return [(output, "exact")]
    return [(output, "exact")]


TOOL = ToolDefinition(
    name="dice_roller",
    description="Roll dice using standard notation (e.g. '2d6', '1d20+3', '4d8-2'). Returns individual rolls and total.",
    inputSchema=DiceRollerInput.model_json_schema(),
    fn=dice_roller,
    extractor=_extractor,
)
