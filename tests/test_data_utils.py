import math

import numpy as np
import pandas as pd

from sommbench.data_utils import build_wine_passage, create_tiered_mask_dataset

TESTABLE_ATTRIBUTES = [
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


class TestCreateTieredMaskDataset:
    def test_deterministic_with_same_seed(self, sample_df: pd.DataFrame) -> None:
        result1 = create_tiered_mask_dataset(sample_df, seed=42)
        result2 = create_tiered_mask_dataset(sample_df, seed=42)
        pd.testing.assert_frame_equal(result1, result2)

    def test_proportions(self, sample_df: pd.DataFrame) -> None:
        result = create_tiered_mask_dataset(sample_df, seed=42)
        n = len(result)
        mask_counts = result[TESTABLE_ATTRIBUTES].apply(
            lambda row: (row == "[MASK]").sum(), axis=1
        )
        n_single = (mask_counts == 1).sum()
        n_double = (mask_counts == 2).sum()
        n_triple = (mask_counts == 3).sum()
        # Proportions should roughly match 40/30/30 (within a few samples)
        assert abs(n_single / n - 0.4) < 0.15
        assert abs(n_double / n - 0.3) < 0.15
        assert abs(n_triple / n - 0.3) < 0.15

    def test_true_columns_contain_originals(self, sample_df: pd.DataFrame) -> None:
        result = create_tiered_mask_dataset(sample_df, seed=42)
        for attr in TESTABLE_ATTRIBUTES:
            true_col = f"true_{attr}"
            assert true_col in result.columns
            # true_ columns should match original values
            for idx in sample_df.index:
                original = sample_df.loc[idx, attr]
                preserved = result.loc[idx, true_col]
                if isinstance(original, list):
                    assert original == preserved
                elif isinstance(original, float) and math.isnan(original):
                    assert math.isnan(preserved)
                else:
                    assert original == preserved

    def test_masked_cells_contain_mask_token(self, sample_df: pd.DataFrame) -> None:
        mask_token = "[MASK]"
        result = create_tiered_mask_dataset(sample_df, mask_token=mask_token, seed=42)
        total_masked = sum(
            (result[attr] == mask_token).sum() for attr in TESTABLE_ATTRIBUTES
        )
        assert total_masked > 0, "No masked cells at all"


class TestBuildWinePassage:
    def test_full_row(self, sample_wfc_row: pd.Series) -> None:
        passage = build_wine_passage(sample_wfc_row)
        assert "Title of wine" in passage
        assert "Type/Color of wine" in passage
        assert "Country of origin" in passage
        assert "Sugar content" in passage
        assert "Alcohol content" in passage

    def test_nan_values_skipped(self, sample_wfc_row: pd.Series) -> None:
        row = sample_wfc_row.copy()
        row["sugar"] = np.nan
        passage = build_wine_passage(row)
        assert "Sugar content" not in passage

    def test_mask_token_skipped(self, sample_wfc_row: pd.Series) -> None:
        row = sample_wfc_row.copy()
        row["country"] = "[MASK]"
        passage = build_wine_passage(row)
        assert "Country of origin" not in passage

    def test_list_grapes_comma_separated(self, sample_wfc_row: pd.Series) -> None:
        passage = build_wine_passage(sample_wfc_row)
        assert "Cabernet Sauvignon, Merlot" in passage

    def test_minimal_row(self) -> None:
        row = pd.Series({"title": "Test Wine"})
        passage = build_wine_passage(row)
        assert "Title of wine: Test Wine" in passage
