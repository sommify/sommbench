import pytest

from sommbench.core import (
    calculate_mape,
    compute_binary_classification_metrics,
    compute_sommbench_score,
    score_wfc_prediction,
)

ALL_ATTRIBUTES = [
    "type",
    "sugar",
    "alcohol",
    "country",
    "region",
    "grapes",
    "dryness",
    "body",
    "acidity",
]


def _make_pred_true_pair() -> tuple[dict, dict]:
    """Return a perfect prediction/true pair for all attributes."""
    pred = {
        "type": "red",
        "sugar": 5.0,
        "alcohol": 13.5,
        "country": "france",
        "region": "bordeaux",
        "grapes": ["Merlot"],
        "dryness": "dry",
        "body": "full bodied",
        "acidity": "medium acidic",
    }
    true = {
        "type": "red",
        "sugar": 5.0,
        "alcohol": 13.5,
        "country": "france",
        "region": ["Bordeaux", "Medoc"],
        "grapes": ["Merlot", "Cabernet Sauvignon"],
        "dryness": "dry",
        "body": "full bodied",
        "acidity": "medium acidic",
    }
    return pred, true


# ---------- calculate_mape ----------


class TestCalculateMape:
    def test_exact_match(self) -> None:
        assert calculate_mape(100, 100) == 0.0

    def test_single_values(self) -> None:
        assert calculate_mape(100, 105) == pytest.approx(5.0)

    def test_lists(self) -> None:
        result = calculate_mape([100, 200], [105, 210])
        assert result == pytest.approx(5.0)

    def test_zero_true_value(self) -> None:
        # sklearn MAPE returns inf for zero true values
        result = calculate_mape(0, 5)
        assert result == float("inf") or result > 1e6


# ---------- score_wfc_prediction ----------


class TestScoreWfcPrediction:
    def test_perfect_prediction_all_masked(self) -> None:
        pred, true = _make_pred_true_pair()
        result = score_wfc_prediction(pred, true, ALL_ATTRIBUTES)
        for attr in ALL_ATTRIBUTES:
            assert result[attr] == 1, f"{attr} should be 1"
        # sugar_mape and alcohol_mape are computed but NOT in masked_attributes
        # so they get set to None
        assert result.get("sugar_mape") is None
        assert result.get("alcohol_mape") is None

    def test_perfect_prediction_only_type_masked(self) -> None:
        pred, true = _make_pred_true_pair()
        result = score_wfc_prediction(pred, true, ["type"])
        assert result["type"] == 1
        # All other attributes should be None (not in masked_attributes)
        for attr in ALL_ATTRIBUTES:
            if attr != "type":
                assert result[attr] is None, f"{attr} should be None"

    def test_type_substring_match(self) -> None:
        pred, true = _make_pred_true_pair()
        # pred "red" is in true "red" → 1
        result = score_wfc_prediction(pred, true, ["type"])
        assert result["type"] == 1

        # pred "sparkling red" is NOT in true "red"
        pred["type"] = "sparkling red"
        result = score_wfc_prediction(pred, true, ["type"])
        assert result["type"] == 0

    def test_type_translation(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["type"] = "rosso"  # Italian for "red"
        true["type"] = "red"
        result = score_wfc_prediction(pred, true, ["type"])
        assert result["type"] == 1

    def test_country_same_english_name(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["country"] = "france"
        true["country"] = "france"
        result = score_wfc_prediction(pred, true, ["country"])
        assert result["country"] == 1

    def test_country_translation_after_fix(self) -> None:
        """After the fix, lowered keys in COUNTRY_TRANSLATIONS_EN_MAP match."""
        pred, true = _make_pred_true_pair()
        pred["country"] = "Deutschland"
        true["country"] = "Germany"
        result = score_wfc_prediction(pred, true, ["country"])
        assert result["country"] == 1

    def test_numeric_within_5_percent(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["sugar"] = 100
        true["sugar"] = 104  # MAPE = 3.85%
        result = score_wfc_prediction(pred, true, ["sugar"])
        assert result["sugar"] == 1
        # sugar_mape is not in masked_attributes so it's None
        assert result.get("sugar_mape") is None

    def test_numeric_outside_5_percent(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["alcohol"] = 10
        true["alcohol"] = 14  # MAPE = 28.6%
        result = score_wfc_prediction(pred, true, ["alcohol"])
        assert result["alcohol"] == 0
        assert result.get("alcohol_mape") is None

    def test_region_membership(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["region"] = "bordeaux"
        true["region"] = ["Bordeaux", "Medoc"]
        result = score_wfc_prediction(pred, true, ["region"])
        assert result["region"] == 1

    def test_grapes_partial_overlap(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["grapes"] = ["Merlot", "Syrah"]
        true["grapes"] = ["Merlot", "Cabernet"]
        result = score_wfc_prediction(pred, true, ["grapes"])
        assert result["grapes"] == 1

    def test_grapes_no_overlap(self) -> None:
        pred, true = _make_pred_true_pair()
        pred["grapes"] = ["Riesling"]
        true["grapes"] = ["Merlot", "Cabernet"]
        result = score_wfc_prediction(pred, true, ["grapes"])
        assert result["grapes"] == 0

    def test_none_true_value(self) -> None:
        pred, true = _make_pred_true_pair()
        true["dryness"] = "none"
        result = score_wfc_prediction(pred, true, ["dryness"])
        assert result["dryness"] == 0


# ---------- compute_binary_classification_metrics ----------


class TestBinaryClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["yes", "yes", "no", "no"]
        m = compute_binary_classification_metrics(y_true, y_pred)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["tnr"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["no", "no", "yes", "yes"]
        m = compute_binary_classification_metrics(y_true, y_pred)
        assert m["mcc"] == pytest.approx(-1.0)
        assert m["tpr"] == pytest.approx(0.0)
        assert m["tnr"] == pytest.approx(0.0)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_all_same_class_positive(self) -> None:
        y_true = ["yes", "yes", "yes"]
        y_pred = ["yes", "yes", "yes"]
        m = compute_binary_classification_metrics(y_true, y_pred)
        # MCC is 0 when one class is absent (sklearn behaviour)
        assert m["mcc"] == pytest.approx(0.0)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_error_outputs_treated_as_negative(self) -> None:
        y_true = ["yes", "no"]
        y_pred = ["error", "error"]
        m = compute_binary_classification_metrics(y_true, y_pred)
        # "error" → negative, so one FN and one TN
        assert m["tnr"] == pytest.approx(1.0)
        assert m["tpr"] == pytest.approx(0.0)

    def test_mixed_performance(self) -> None:
        y_true = ["yes", "yes", "no", "no", "yes", "no"]
        y_pred = ["yes", "no", "no", "yes", "yes", "no"]
        m = compute_binary_classification_metrics(y_true, y_pred)
        # TP=2, FN=1, TN=2, FP=1
        assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-3)
        assert m["tpr"] == pytest.approx(2 / 3, abs=1e-3)
        assert m["tnr"] == pytest.approx(2 / 3, abs=1e-3)


# ---------- compute_sommbench_score ----------


class TestSommBenchScore:
    def test_basic_arithmetic(self) -> None:
        wtqa = {"aggregated_results": {"overall": {"accuracy": 0.6}}}
        fwp = {"aggregated_results": {"mcc": 0.3}}
        wfc = {"aggregated_results": {"overall": {"score": 0.9}}}
        result = compute_sommbench_score(wtqa, fwp, wfc)
        expected = round((0.6 + 0.3 + 0.9) / 3, 4)
        assert result["sommbench_score"] == pytest.approx(expected)
        assert result["s_wtqa"] == pytest.approx(0.6)
        assert result["s_fwp"] == pytest.approx(0.3)
        assert result["s_wfc"] == pytest.approx(0.9)

    def test_all_perfect(self) -> None:
        wtqa = {"aggregated_results": {"overall": {"accuracy": 1.0}}}
        fwp = {"aggregated_results": {"mcc": 1.0}}
        wfc = {"aggregated_results": {"overall": {"score": 1.0}}}
        result = compute_sommbench_score(wtqa, fwp, wfc)
        assert result["sommbench_score"] == pytest.approx(1.0)

    def test_all_zero(self) -> None:
        wtqa = {"aggregated_results": {"overall": {"accuracy": 0.0}}}
        fwp = {"aggregated_results": {"mcc": 0.0}}
        wfc = {"aggregated_results": {"overall": {"score": 0.0}}}
        result = compute_sommbench_score(wtqa, fwp, wfc)
        assert result["sommbench_score"] == pytest.approx(0.0)
