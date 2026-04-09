from unittest.mock import MagicMock, patch

from sommbench.core import _extract_usage, _run_batch_completion

# ---------- _extract_usage ----------


class TestExtractUsage:
    def test_valid_response(self) -> None:
        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.total_tokens = 30
        assert _extract_usage(response) == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_none_response(self) -> None:
        assert _extract_usage(None) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_missing_usage_attr(self) -> None:
        response = MagicMock(spec=[])  # no attributes
        assert _extract_usage(response) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


# ---------- _run_batch_completion ----------


def _make_mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 10
    resp.usage.total_tokens = 15
    return resp


class TestRunBatchCompletion:
    """Test _run_batch_completion with batch_size=1, small, and larger batch sizes."""

    @patch("sommbench.core.litellm")
    def test_batch_size_1(self, mock_litellm: MagicMock) -> None:
        """batch_size=1 should call litellm.completion once per message."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
        mock_litellm.completion.return_value = _make_mock_response("answer")

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=1
        )

        assert len(responses) == 3
        assert mock_litellm.completion.call_count == 3
        mock_litellm.batch_completion.assert_not_called()

    @patch("sommbench.core.litellm")
    def test_batch_size_matches_messages(self, mock_litellm: MagicMock) -> None:
        """batch_size == len(messages) should make exactly 1 batch_completion call."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
        mock_responses = [_make_mock_response(f"a{i}") for i in range(4)]
        mock_litellm.batch_completion.return_value = mock_responses

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=4
        )

        assert len(responses) == 4
        assert mock_litellm.batch_completion.call_count == 1
        mock_litellm.completion.assert_not_called()

    @patch("sommbench.core.litellm")
    def test_batch_size_larger_than_messages(self, mock_litellm: MagicMock) -> None:
        """batch_size > len(messages) should still work (1 batch with all messages)."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
        mock_responses = [_make_mock_response(f"a{i}") for i in range(3)]
        mock_litellm.batch_completion.return_value = mock_responses

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=100
        )

        assert len(responses) == 3
        assert mock_litellm.batch_completion.call_count == 1

    @patch("sommbench.core.litellm")
    def test_batch_size_splits_evenly(self, mock_litellm: MagicMock) -> None:
        """10 messages with batch_size=5 should make exactly 2 batch_completion calls."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(10)]
        mock_litellm.batch_completion.return_value = [
            _make_mock_response(f"a{i}") for i in range(5)
        ]

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=5
        )

        assert len(responses) == 10
        assert mock_litellm.batch_completion.call_count == 2

    @patch("sommbench.core.litellm")
    def test_batch_size_splits_unevenly(self, mock_litellm: MagicMock) -> None:
        """7 messages with batch_size=3 should make 3 calls (3+3+1)."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(7)]

        def side_effect(**kwargs) -> list[MagicMock]:
            batch = kwargs["messages"]
            return [_make_mock_response(f"a{i}") for i in range(len(batch))]

        mock_litellm.batch_completion.side_effect = side_effect

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=3
        )

        assert len(responses) == 7
        assert mock_litellm.batch_completion.call_count == 3

    @patch("sommbench.core.litellm")
    def test_error_handling_batch_size_1(self, mock_litellm: MagicMock) -> None:
        """On error with batch_size=1, should append None for that item."""
        messages = [[{"role": "user", "content": "q0"}]]
        mock_litellm.completion.side_effect = RuntimeError("API down")

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=1
        )

        assert len(responses) == 1
        assert responses[0] is None

    @patch("sommbench.core.litellm")
    def test_error_handling_batch_size_gt1(self, mock_litellm: MagicMock) -> None:
        """On error with batch_size>1, should extend with None * len(batch)."""
        messages = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
        mock_litellm.batch_completion.side_effect = RuntimeError("API down")

        responses = _run_batch_completion(
            messages, "model", "key", "http://base", batch_size=4
        )

        assert len(responses) == 4
        assert all(r is None for r in responses)

    @patch("sommbench.core.litellm")
    def test_kwargs_forwarded(self, mock_litellm: MagicMock) -> None:
        """Extra kwargs (e.g. response_format) should be passed through."""
        messages = [[{"role": "user", "content": "q0"}]]
        mock_litellm.completion.return_value = _make_mock_response("a0")
        fake_schema = MagicMock()

        _run_batch_completion(
            messages,
            "model",
            "key",
            "http://base",
            batch_size=1,
            response_format=fake_schema,
            temperature=0.5,
        )

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["response_format"] is fake_schema
        assert call_kwargs.kwargs["temperature"] == 0.5
