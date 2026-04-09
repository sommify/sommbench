import random

import numpy as np
import pandas as pd


def create_tiered_mask_dataset(
    df: pd.DataFrame, mask_token: str = "[MASK]", seed: int = 42
) -> pd.DataFrame:
    random.seed(seed)
    df_masked = df.copy()

    n_samples = len(df)
    proportions = {"single": 0.4, "double": 0.30, "triple": 0.30}

    testable_attributes = [
        "type",
        "sugar",
        "alcohol",
        "country",
        "region",
        "grapes",
        "dryness",
        "acidity",
        "body",
    ]
    for col in testable_attributes:
        df_masked[col] = df_masked[col].astype(object)

    for col in testable_attributes:
        df_masked[f"true_{col}"] = df_masked[col]

    double_mask_pairs = [
        ["region", "country"],  # Geo-Identity
        ["type", "grapes"],  # Varietal-Type Link
        ["type", "region"],
        ["sugar", "alcohol"],  # Technical Profile
    ]
    triple_mask_triplets = [
        ["country", "region", "type"],  # Core Identity Crisis
        ["alcohol", "sugar", "grapes"],  # Complete Technical Profile
        ["dryness", "acidity", "body"],  # Winemaking Style Puzzle
    ]

    indices = df.index.tolist()
    random.shuffle(indices)

    n_single = int(n_samples * proportions["single"])
    n_double = int(n_samples * proportions["double"])
    # The rest are triple

    single_mask_indices = indices[:n_single]
    double_mask_indices = indices[n_single : n_single + n_double]
    triple_mask_indices = indices[n_single + n_double :]

    num_attributes = len(testable_attributes)
    cols_to_mask_single = (testable_attributes * (n_single // num_attributes + 1))[
        :n_single
    ]
    random.shuffle(cols_to_mask_single)

    for i, col in zip(single_mask_indices, cols_to_mask_single, strict=False):
        df_masked.loc[i, col] = mask_token

    # Tier 2: Double-Mask Entries (~30%)
    for i in double_mask_indices:
        pair = random.choice(double_mask_pairs)
        for col in pair:
            df_masked.loc[i, col] = mask_token

    for i in triple_mask_indices:
        triplet = random.choice(triple_mask_triplets)
        for col in triplet:
            df_masked.loc[i, col] = mask_token

    return df_masked


def build_wine_passage(row: pd.Series, mask_token: str = "[MASK]") -> str:
    def _format_value(value: str | list[str]) -> str:
        if isinstance(value, list | np.ndarray):
            return ", ".join(str(v) for v in value)
        return str(value)

    wine_attributes = {
        "title": ("Title of wine", ""),
        "type": ("Type/Color of wine", ""),
        "grapes": ("Grapes from which the wine was made", ""),
        "country": ("Country of origin", ""),
        "region": ("Region of origin", ""),
        "sugar": ("Sugar content", "g/L"),
        "alcohol": ("Alcohol content of wine", "%"),
        "dryness": ("Dry-Sweet taste of wine", ""),
        "acidity": ("Acidic taste of wine", ""),
        "body": ("Taste of body in the wine", ""),
    }
    components = []
    for col, (prefix, suffix) in wine_attributes.items():
        value = row.get(col)
        if not isinstance(value, list | np.ndarray) and pd.isna(value):
            continue
        if isinstance(value, list | np.ndarray) or (value and value != mask_token):
            formatted_value = _format_value(value)
            if not pd.isna(formatted_value):
                components.append(f"{prefix}: {formatted_value}{suffix}")
    return "; ".join(components)
