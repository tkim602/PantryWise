"""Ingredient cleaning utilities for the recipe datasets."""

from __future__ import annotations

import ast
import re
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd


DATA_REL_PATH = (
    "data/food-ingredients-and-recipe-dataset-with-images/"
    "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
)

UNITS = [
    "tsp", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons",
    "cup", "cups", "pint", "pints", "quart", "quarts", "gallon", "gallons",
    "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gram",
    "grams", "kg", "kilogram", "kilograms", "ml", "milliliter",
    "milliliters", "l", "liter", "liters", "stick", "sticks", "slice",
    "slices", "clove", "cloves", "head", "heads", "piece", "pieces",
    "package", "packages", "can", "cans",
]

PHRASES_TO_DROP = [
    r"to taste",
    r"plus more",
    r"or more",
    r"for serving",
    r"to serve",
    r"per person",
    r"divided",
    r"room temperature",
]

PREP_WORDS = [
    "split", "halved", "quartered", "chopped", "diced", "minced", "grated",
    "shredded", "peeled", "seeded", "cored", "rinsed", "drained", "melted",
    "softened", "sifted", "sliced", "thinly", "thickly", "finely", "coarsely",
    "roughly", "freshly", "ground",
]


def _strip_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def _remove_unicode_fractions(text: str) -> str:
    return re.sub(r"[½¼¾⅓⅔⅛⅜⅝⅞⁄]", " ", text)


def parse_ingredient_quantity(raw: str) -> dict:
    """Parse an ingredient string into quantity, unit, and normalized name."""
    if not isinstance(raw, str) or not raw.strip():
        return {"quantity": 1.0, "unit": "", "name": ""}

    text = _remove_unicode_fractions(_strip_parentheses(raw.strip().lower()))
    pattern = (
        r"^(\d+(?:\s+\d+/\d+)?(?:\.\d+)?)?\s*"
        r"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)?\s*(.*)$"
    )
    match = re.match(pattern, text)
    if not match:
        return {"quantity": 1.0, "unit": "", "name": normalize_ingredient(raw)}

    qty_str, unit_str, name_str = match.groups()
    qty_str = qty_str or "1"
    unit_str = unit_str or ""
    name_str = name_str or ""

    if not name_str.strip():
        remainder = text
        if qty_str:
            remainder = re.sub(r"^\s*" + re.escape(qty_str), "", remainder)
        if unit_str:
            remainder = re.sub(r"^\s*" + re.escape(unit_str), "", remainder)
        name_str = remainder.strip()

    try:
        quantity = float(Fraction(qty_str)) if "/" in qty_str else float(qty_str)
    except ValueError:
        quantity = 1.0

    unit = ""
    unit_candidate = unit_str.strip()
    if unit_candidate:
        tokens = unit_candidate.split()
        if tokens[0] in UNITS:
            unit = tokens[0]
            leftover = " ".join(tokens[1:]).strip()
            if leftover:
                name_str = f"{leftover} {name_str}".strip()
        elif not name_str.startswith(unit_candidate):
            name_str = f"{unit_candidate} {name_str}".strip()

    return {
        "quantity": quantity,
        "unit": unit,
        "name": normalize_ingredient(name_str),
    }


def normalize_ingredient(raw: str) -> str:
    """Normalize an ingredient string to a name-only representation."""
    if not isinstance(raw, str):
        return ""

    text = _remove_unicode_fractions(_strip_parentheses(raw.strip().lower()))

    text = re.sub(
        r"\b\d+(?:[\./]\d+)?\s+to\s+\d+(?:[\./]\d+)?\b",
        " ",
        text,
    )

    units_group = "|".join(map(re.escape, UNITS))
    text = re.sub(
        rf"\b\d+(?:[\./]\d+)?\s*(?:{units_group})\.?(?!\w)",
        " ",
        text,
    )
    text = re.sub(r"\b\d+\s*[\-–—]\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+(?:[\./]\d+)?\b", " ", text)
    text = re.sub(r"\b[x×]\b", " ", text)

    units_pattern = r"\b(" + units_group + r")\b\.?"
    text = re.sub(units_pattern, " ", text)

    for phrase in PHRASES_TO_DROP:
        text = re.sub(rf"\b{phrase}\b", " ", text)

    text = re.sub(r"^\s*(?:or|and)\s+", "", text)

    prep_pattern = r"\b(" + "|".join(map(re.escape, PREP_WORDS)) + r")\b"
    text = re.sub(prep_pattern, " ", text)
    text = re.sub(r"\b(pinch|dash)\s+of\b", " ", text)
    text = re.sub(r"[,/%:;\-–—]", " ", text)
    text = text.replace('"', " ").replace("'", " ")
    text = re.sub(r"^of\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_ingredient_list(cell: object) -> list[str]:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell:
        return []

    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return [part.strip() for part in cell.split(",") if part.strip()]

    if isinstance(value, list):
        return value
    return [str(value)]


def _normalize_ingredient_list(items: list[str]) -> list[str]:
    normalized = []
    for item in items:
        name = normalize_ingredient(item)
        if name:
            normalized.append(name)
    return normalized


def normalize_ingredients_column(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return normalized ingredient names and parsed quantity details."""
    if "Cleaned_Ingredients" in df.columns:
        source_col = "Cleaned_Ingredients"
    elif "Ingredients" in df.columns:
        source_col = "Ingredients"
    else:
        raise KeyError("No 'Cleaned_Ingredients' or 'Ingredients' column found in CSV")

    parsed = df[source_col].apply(_parse_ingredient_list)
    normalized = parsed.apply(_normalize_ingredient_list)
    parsed_details = parsed.apply(
        lambda items: [
            detail
            for detail in (parse_ingredient_quantity(item) for item in items)
            if detail.get("name")
        ]
    )
    return normalized, parsed_details


def main(argv: list[str]) -> int:
    in_path = Path(argv[1]) if len(argv) > 1 else Path(DATA_REL_PATH)
    out_path = (
        Path(argv[2])
        if len(argv) > 2
        else in_path.with_name(f"{in_path.stem}.normalized.csv")
    )

    project_root = Path(__file__).resolve().parent
    in_abs = (project_root / in_path).resolve()
    out_abs = (project_root / out_path).resolve()
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_abs}")
    df = pd.read_csv(in_abs)
    df["Normalized_Ingredients"], df["Parsed_Ingredients"] = (
        normalize_ingredients_column(df)
    )

    print(f"Writing normalized CSV: {out_abs}")
    df.to_csv(out_abs, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
