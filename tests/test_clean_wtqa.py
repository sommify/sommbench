import pytest

from sommbench.core import clean_wtqa_answer


@pytest.mark.parametrize(
    ("raw", "lang", "expected"),
    [
        # Special token stripping
        ("a<|im_end|>", "en", "a"),
        ("<|im_start|>b<|im_end|>", "en", "b"),
        ("c<|endoftext|>", "en", "c"),
        # No special tokens (passthrough)
        ("a", "en", "a"),
        # Italian prefix removal
        ("la risposta è b", "it", "b"),
        # Spanish prefix removal
        ("la respuesta es: c", "es", "c"),
        # Finnish prefix removal
        ("oikea vastaus on d", "fi", "d"),
        # German prefix removal
        ("die richtige antwort ist: a", "de", "a"),
        # Parenthesized extraction
        ("(a) some text", "en", "a"),
        ("(d)", "en", "d"),
        # Combined: special token + language prefix
        ("la risposta è a<|im_end|>", "it", "a"),
        # Already clean letter
        ("b", "de", "b"),
    ],
)
def test_clean_wtqa_answer(raw: str, lang: str, expected: str) -> None:
    assert clean_wtqa_answer(raw, lang) == expected
