import inspect

from typer.testing import CliRunner

from sommbench.cli import app
from sommbench.core import run_fwp_benchmark, run_wfc_benchmark, run_wtqa_benchmark

runner = CliRunner()


def test_fwp_help_no_language_option() -> None:
    result = runner.invoke(app, ["fwp", "--help"])
    assert result.exit_code == 0
    assert "--language" not in result.output


def test_run_help_has_language_option() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--language" in result.output


def test_fwp_signature_no_language() -> None:
    sig = inspect.signature(run_fwp_benchmark)
    assert "language" not in sig.parameters


def test_no_think_flag_on_all_commands() -> None:
    for cmd in ["wtqa", "fwp", "wfc", "run"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "--no-think" in result.output, f"--no-think missing from {cmd}"


def test_no_think_parameter_in_core_signatures() -> None:
    for func in [run_wtqa_benchmark, run_fwp_benchmark, run_wfc_benchmark]:
        sig = inspect.signature(func)
        assert "no_think" in sig.parameters, f"no_think missing from {func.__name__}"
        param = sig.parameters["no_think"]
        assert param.default is False
