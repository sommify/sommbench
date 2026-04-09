"""SommBench: A multilingual wine sommelier LLM benchmark."""

from importlib.metadata import version

__version__ = version("sommbench")

from sommbench.cli import run
from sommbench.core import (
    compute_sommbench_score,
    run_fwp_benchmark,
    run_wfc_benchmark,
    run_wtqa_benchmark,
)
