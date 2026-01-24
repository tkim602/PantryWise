import pandas as pd
from pathlib import Path
from utils import parse_list

TAG_RULES = {
    "meat": ["chicken", "beef", "pork", "lamb", "bacon", "fish", "shrimp", "salmon", "turkey"],
    "vegetarian": ["tofu", "mushroom", "lentil", "beans", "broccoli", "spinach"],
    "dessert": ["sugar", "flour", "butter", "chocolate", "apple", "vanilla"],
    "dairy": ["milk", "cheese", "cream", "butter", "yogurt", "cream cheese"],
}

DATA_REL_PATH = (
    "data/food-ingredients-and-recipe-dataset-with-images/versions/1/"
    "Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
)


def infer_tags(ingredients_list):
    tags = set()
    for tag, keywords in TAG_RULES.items():
        for ing in ingredients_list:
            if any(k in ing.lower() for k in keywords):
                tags.add(tag)
    return list(tags)


def generate_tags(in_path=DATA_REL_PATH, out_path=None):
    """Load normalized CSV, infer tags, and one-hot encode them."""
    project_root = Path(__file__).resolve().parent
    in_abs = (project_root / in_path).resolve()
    print(f"Reading: {in_abs}")
    df = pd.read_csv(in_abs)

    # Choose best available ingredients column
    for col in ["Normalized_Ingredients", "Cleaned_Ingredients", "Ingredients"]:
        if col in df.columns:
            source_col = col
            break
    else:
        raise KeyError("No ingredient column found in CSV")

    df[source_col] = df[source_col].apply(parse_list)

    # Infer and encode tags
    df["Tags"] = df[source_col].apply(infer_tags)
    tag_dummies = df["Tags"].str.join("|").str.get_dummies()
    df = df.join(tag_dummies)

    # Save file
    if out_path is None:
        out_path = str(in_abs).replace(".csv", ".with_tags.csv")

    out_abs = Path(out_path)
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_abs, index=False)
    print(f"✅ Saved file with inferred tags at: {out_abs}")


if __name__ == "__main__":
    generate_tags()
