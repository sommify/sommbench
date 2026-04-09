import pytest

from sommbench.core import run_fwp_benchmark, run_wtqa_benchmark


@pytest.mark.integration
class TestWtqaIntegration:
    def test_wtqa_benchmark(self, integration_model_config: dict) -> None:
        result = run_wtqa_benchmark(
            model=integration_model_config["model"],
            language="en",
            num_passes=1,
            api_key=integration_model_config["api_key"],
            api_base=integration_model_config["api_base"],
            batch_size=1,
            sample_size=5,
        )
        assert "run_metadata" in result
        assert "aggregated_results" in result
        assert "instances" in result

        # Check that LLM outputs are clean single letters
        for lang_instances in result["instances"].values():
            for iteration in lang_instances:
                for inst in iteration:
                    if inst["llm_output"] is not None:
                        assert "<|" not in inst["llm_output"]

        # Accuracy should be a float between 0 and 1
        for _lang_key, agg in result["aggregated_results"].items():
            assert 0 <= agg["accuracy"] <= 1


@pytest.mark.integration
class TestFwpIntegration:
    def test_fwp_benchmark(self, integration_model_config: dict) -> None:
        result = run_fwp_benchmark(
            model=integration_model_config["model"],
            num_passes=1,
            api_key=integration_model_config["api_key"],
            api_base=integration_model_config["api_base"],
            batch_size=1,
            sample_size=5,
        )
        assert "run_metadata" in result
        assert "aggregated_results" in result
        assert "instances" in result
