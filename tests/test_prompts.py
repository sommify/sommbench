import pytest

from sommbench.prompts import (
    build_fwp_prompt,
    build_wfc_prompt,
    build_wtqa_prompt,
    get_wfc_schema,
)


class TestBuildWtqaPrompt:
    @pytest.mark.parametrize("lang", ["en", "sk", "de", "da", "it", "es", "fi", "sv"])
    def test_contains_question_and_options(self, lang: str) -> None:
        prompt = build_wtqa_prompt(
            question="What grape?",
            a="Merlot",
            b="Syrah",
            c="Riesling",
            d="Chardonnay",
            lang=lang,
        )
        assert "What grape?" in prompt
        assert "Merlot" in prompt
        assert "Syrah" in prompt
        assert "Riesling" in prompt
        assert "Chardonnay" in prompt
        assert "(A)" in prompt
        assert "(D)" in prompt


class TestBuildFwpPrompt:
    def test_contains_recipe_and_wine(self) -> None:
        prompt = build_fwp_prompt(recipe="Grilled salmon", wine="Pinot Noir")
        assert "Grilled salmon" in prompt
        assert "Pinot Noir" in prompt


class TestBuildWfcPrompt:
    @pytest.mark.parametrize("lang", ["en", "sk", "da", "de", "it", "es", "fi", "sv"])
    def test_contains_passage(self, lang: str) -> None:
        prompt = build_wfc_prompt(passage="A fine French red wine", language=lang)
        assert "A fine French red wine" in prompt

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            build_wfc_prompt(passage="test", language="xx")


class TestGetWfcSchema:
    @pytest.mark.parametrize("lang", ["en", "sk", "da", "de", "it", "es", "fi", "sv"])
    def test_returns_pydantic_class(self, lang: str) -> None:
        schema = get_wfc_schema(lang)
        assert hasattr(schema, "model_json_schema")

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            get_wfc_schema("xx")
