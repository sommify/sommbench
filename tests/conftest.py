from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_wfc_row() -> pd.Series:
    return pd.Series(
        {
            "title": "Chateau Margaux 2015",
            "type": "red",
            "sugar": 2.5,
            "alcohol": 13.5,
            "country": "France",
            "region": ["Bordeaux", "Medoc"],
            "grapes": ["Cabernet Sauvignon", "Merlot"],
            "dryness": "dry",
            "body": "full bodied",
            "acidity": "medium acidic",
        }
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 20
    types = ["red", "white", "rose"]
    countries = ["France", "Italy", "Spain", "Germany"]
    dryness_vals = ["dry", "medium dry", "sweet"]
    body_vals = ["light bodied", "medium bodied", "full bodied"]
    acidity_vals = ["slightly acidic", "medium acidic", "acidity"]

    return pd.DataFrame(
        {
            "title": [f"Wine {i}" for i in range(n)],
            "type": rng.choice(types, n).tolist(),
            "sugar": rng.uniform(0, 50, n).tolist(),
            "alcohol": rng.uniform(8, 16, n).tolist(),
            "country": rng.choice(countries, n).tolist(),
            "region": [["Region A", "Region B"] for _ in range(n)],
            "grapes": [["Grape A", "Grape B"] for _ in range(n)],
            "dryness": rng.choice(dryness_vals, n).tolist(),
            "body": rng.choice(body_vals, n).tolist(),
            "acidity": rng.choice(acidity_vals, n).tolist(),
        }
    )


class MockMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class MockChoice:
    def __init__(self, content: object) -> None:
        self.message = MockMessage(content)


class MockResponse:
    def __init__(self, content: object) -> None:
        self.choices = [MockChoice(content)]


def make_mock_response(content: object) -> MockResponse:
    return MockResponse(content)


@pytest.fixture()
def mock_response() -> Callable[[Any], MockResponse]:
    """Return the factory function itself so tests can call it."""
    return make_mock_response


NULL_DICT = {
    "type": None,
    "sugar": None,
    "alcohol": None,
    "country": None,
    "region": None,
    "grapes": None,
    "dryness": None,
    "body": None,
    "acidity": None,
}


@pytest.fixture()
def null_dict() -> dict:
    return NULL_DICT.copy()


@pytest.fixture()
def integration_model_config() -> dict:
    return {
        "model": "openai/qwen3.5-0.8b-mlx",
        "api_base": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    }
