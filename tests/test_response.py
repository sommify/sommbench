import json

from sommbench.core import get_dict_from_response
from sommbench.prompts import EN_WineSchema


class TestGetDictFromResponse:
    def test_none_response(self, null_dict: dict) -> None:
        result = get_dict_from_response(None, EN_WineSchema)
        assert result == null_dict

    def test_none_content(self, mock_response: type, null_dict: dict) -> None:
        resp = mock_response(None)
        result = get_dict_from_response(resp, EN_WineSchema)
        assert result == null_dict

    def test_valid_json_string(self, mock_response: type) -> None:
        data = {
            "type": "red",
            "sugar": 5.0,
            "alcohol": 13.5,
            "country": "France",
            "region": "Bordeaux",
            "grapes": ["Merlot"],
            "dryness": "dry",
            "body": "full bodied",
            "acidity": "medium acidic",
        }
        resp = mock_response(json.dumps(data))
        result = get_dict_from_response(resp, EN_WineSchema)
        assert result["type"] == "red"
        assert result["sugar"] == 5.0
        assert result["grapes"] == ["Merlot"]

    def test_pydantic_model_instance(self, mock_response: type) -> None:
        instance = EN_WineSchema(
            type="white",
            sugar=3.0,
            alcohol=12.0,
            country="Italy",
            region="Tuscany",
            grapes=["Sangiovese"],
            dryness="dry",
            body="medium bodied",
            acidity="medium acidic",
        )
        resp = mock_response(instance)
        result = get_dict_from_response(resp, EN_WineSchema)
        assert result["type"] == "white"
        assert result["country"] == "Italy"

    def test_missing_keys_filled_with_none(self, mock_response: type) -> None:
        data = {"type": "red", "sugar": 5.0}
        resp = mock_response(json.dumps(data))
        result = get_dict_from_response(resp, EN_WineSchema)
        assert result["type"] == "red"
        assert result["alcohol"] is None
        assert result["grapes"] is None

    def test_malformed_json(self, mock_response: type, null_dict: dict) -> None:
        resp = mock_response("not valid json {{{")
        result = get_dict_from_response(resp, EN_WineSchema)
        assert result == null_dict
