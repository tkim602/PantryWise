import ast
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd


DATA_REL_PATH = (
	"data/food-ingredients-and-recipe-dataset-with-images/"
	"Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
)


def _strip_parentheses(text: str) -> str:
	return re.sub(r"\([^\)]*\)", " ", text)


def _remove_unicode_fractions(text: str) -> str:
	# also remove slash from fracs
	return re.sub(r"[½¼¾⅓⅔⅛⅜⅝⅞⁄]", " ", text)


UNITS = [
	"tsp", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons",
	"cup", "cups", "pint", "pints", "quart", "quarts", "gallon", "gallons",
	"oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gram", "grams", "kg", "kilogram", "kilograms",
	"ml", "milliliter", "milliliters", "l", "liter", "liters",
	"stick", "sticks", "slice", "slices", "clove", "cloves", "head", "heads",
	"piece", "pieces", "package", "packages", "can", "cans",
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


# preparation/ descriptive words to drop
PREP_WORDS = [
	"split", "halved", "quartered", "chopped", "diced", "minced",
	"grated", "shredded", "peeled", "seeded", "cored", "rinsed",
	"drained", "melted", "softened", "sifted",
	"sliced", "thinly", "thickly",
	"finely", "coarsely", "roughly", "freshly", "ground",
]


def parse_ingredient_quantity(raw: str) -> dict:
	"""
	raw ingredient string -> quantity, unit, and name
	Returns {'quantity': float, 'unit': str, 'name': str}
	"""
	if not isinstance(raw, str) or not raw.strip():
		return {'quantity': 1.0, 'unit': '', 'name': ''}

	s = raw.strip().lower()
	s = _strip_parentheses(s)
	s = _remove_unicode_fractions(s)

	# Regex to capture quantity (number or fraction), unit, and name
	pattern = r'^(\d+(?:\s+\d+/\d+)?(?:\.\d+)?)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)?\s*(.*)$'
	match = re.match(pattern, s)
	if not match:
		return {'quantity': 1.0, 'unit': '', 'name': normalize_ingredient(raw)}

	qty_str, unit_str, name_str = match.groups()
	qty_str = qty_str or '1'
	unit_str = unit_str or ''
	name_str = name_str or ''

	if not name_str.strip():
		remainder = s
		if qty_str:
			remainder = re.sub(r'^\s*' + re.escape(qty_str), '', remainder)
		if unit_str:
			remainder = re.sub(r'^\s*' + re.escape(unit_str), '', remainder)
		name_str = remainder.strip() or ''

	try:
		if '/' in qty_str:
			qty = float(Fraction(qty_str))
		else:
			qty = float(qty_str)
	except ValueError:
		qty = 1.0

	unit_candidate = unit_str.strip()
	unit = ""
	if unit_candidate:
		tokens = unit_candidate.split()
		if tokens[0] in UNITS:
			unit = tokens[0]
			leftover = " ".join(tokens[1:]).strip()
			if leftover:
				name_str = (leftover + " " + name_str).strip() if name_str else leftover
		else:
			if not name_str.startswith(unit_candidate):
				name_str = (unit_candidate + " " + name_str).strip()

	name = normalize_ingredient(name_str)

	return {'quantity': qty, 'unit': unit, 'name': name}


def normalize_ingredient(raw: str) -> str:
	"""
	- lowercase
	- remove numbers, measures, quantity phrases (like to taste), parens
	- punctuation + whitepsace
	"""

	if not isinstance(raw, str):
		return ""

	s = raw.strip().lower()
	s = _strip_parentheses(s)
	s = _remove_unicode_fractions(s)

	# remove ranges with words like "15 to 20"
	s = re.sub(r"\b\d+(?:[\./]\d+)?\s+to\s+\d+(?:[\./]\d+)?\b", " ", s)

	# remove number+unit tokens like "500g"
	units_group = "|".join(map(re.escape, UNITS))
	s = re.sub(rf"\b\d+(?:[\./]\d+)?\s*(?:{units_group})\.?(?!\w)", " ", s)

	# remove ranges
	s = re.sub(r"\b\d+\s*[\-–—]\s*\d+\b", " ", s)

	# remove numbers
	s = re.sub(r"\b\d+(?:[\./]\d+)?\b", " ", s)

	# remove multiplyers like x
	s = re.sub(r"\b[x×]\b", " ", s)

	# remove measurement units
	units_pattern = r"\b(" + "|".join(map(re.escape, UNITS)) + r")\b\.?"
	s = re.sub(units_pattern, " ", s)

	# remove quantity phrases
	for phrase in PHRASES_TO_DROP:
		s = re.sub(rf"\b{phrase}\b", " ", s)

	# remove leading conjunctions
	s = re.sub(r"^\s*(?:or|and)\s+", "", s)

	# remove preparation/descriptive words
	prep_pattern = r"\b(" + "|".join(map(re.escape, PREP_WORDS)) + r")\b"
	s = re.sub(prep_pattern, " ", s)

	# remove pinch and dash of
	s = re.sub(r"\b(pinch|dash)\s+of\b", " ", s)

	# remove extra punctuation (also includes / now)
	s = re.sub(r"[,/%:;\-–—]", " ", s)

	# remove stray quotes
	s = s.replace('"', ' ').replace("'", " ")

	# remove leading ifs
	s = re.sub(r"^of\s+", "", s)

	# remove whitespace
	s = re.sub(r"\s+", " ", s).strip()

	return s


def normalize_ingredients_column(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Returns two Series: normalized names and parsed details."""
	source_col = None
	if "Cleaned_Ingredients" in df.columns:
		source_col = "Cleaned_Ingredients"
	elif "Ingredients" in df.columns:
		source_col = "Ingredients"
	else:
		raise KeyError("No 'Cleaned_Ingredients' or 'Ingredients' column found in CSV")

	def parse_list(cell):
		if isinstance(cell, list):
			return cell
		if not isinstance(cell, str) or not cell:
			return []
		try:
			value = ast.literal_eval(cell)
			if isinstance(value, list):
				return value
			return [str(value)]
		except Exception:
			return [c.strip() for c in str(cell).split(",") if c.strip()]

	parsed = df[source_col].apply(parse_list)
	normalized = parsed.apply(lambda lst: [normalize_ingredient(x) for x in lst if normalize_ingredient(x)])

	parsed_details = parsed.apply(
		lambda lst: [d for d in (parse_ingredient_quantity(x) for x in lst) if d and d.get('name')]
	)

	return normalized, parsed_details


def main(argv: list[str]) -> int:
	in_path = Path(argv[1]) if len(argv) > 1 else Path(DATA_REL_PATH)
	out_path = (
		Path(argv[2])
		if len(argv) > 2
		else in_path.with_name(in_path.stem + ".normalized.csv")
	)

	project_root = Path(__file__).resolve().parent
	in_abs = (project_root / in_path).resolve()
	out_abs = (project_root / out_path).resolve()
	out_abs.parent.mkdir(parents=True, exist_ok=True)

	print(f"Reading: {in_abs}")
	df = pd.read_csv(in_abs)

	df["Normalized_Ingredients"], df["Parsed_Ingredients"] = normalize_ingredients_column(df)

	print(f"Writing normalized CSV: {out_abs}")
	df.to_csv(out_abs, index=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))