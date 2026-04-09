# SommBench: Assessing Sommelier Expertise of Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.12117-b31b1b.svg)](https://arxiv.org/abs/2603.12117)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**SommBench** is a multilingual benchmark for evaluating sommelier-level wine knowledge in large language models. It spans three complementary tasks — Wine Theory Q&A, Food & Wine Pairing, and Wine Feature Completion.

## Quick Start

```sh
uv add sommbench
```

Create a `.env` file with your API credentials:

```env
API_KEY="your-api-key"
API_BASE="https://api.openai.com/v1"
```

Run the full benchmark suite:

```sh
sommbench run "lm_studio/qwen3.5-0.8b-mlx"
```


## Usage

### Command Line

`sommbench <COMMAND> <MODEL> [OPTIONS]`

#### Commands

| Command | Description |
| :--- | :--- |
| `run` | Runs **all three** benchmarks and prints the composite SommBench Score |
| `wtqa` | **Wine Theory Question-Answering** — multiple-choice wine knowledge quiz |
| `fwp` | **Food & Wine Pairing** — predicts whether a wine pairs with a dish (English only) |
| `wfc` | **Wine Feature Completion** — fills in masked wine attributes via structured output |

#### Examples

**Run all benchmarks:**

```sh
sommbench run "gpt-4o" --output ./benchmark_results/
```

**Single task — WTQA on all languages:**

```sh
sommbench wtqa "gpt-4o"
```

**Specify languages for WFC (Italian and German):**

```sh
sommbench wfc "gemini-1.5-pro" -l it -l de --batch-size 64 --output wfc_results.json
```

**Multiple passes for FWP:**

```sh
sommbench fwp "claude-sonnet-4-6" --num-passes 3 --output fwp_results.json
```

**Extra model parameters:**

```sh
sommbench run "lm_studio/my-model" --model-params '{"stop": ["<|im_end|>"], "max_tokens": 512}'
```

**Inline credentials:**

```sh
sommbench wtqa "gpt-4-turbo" --api-key "sk-..." --api-base "https://api.openai.com/v1"
```

#### Options

```sh
sommmbech --help
```

| Option | Description | Default |
| :--- | :--- | :--- |
| `model` (positional) | **Required.** Model name/identifier passed to LiteLLM. | — |
| `--output <PATH>` | Output file (or directory) for results. | `results.json` (`results/` for `run`) |
| `-l`, `--language <LANG>` | Language(s) to benchmark. Repeat for multiple. Not available for `fwp`. | `all` |
| `--num-passes <INT>` | Number of full benchmark passes (each pass evaluates every item once). | `1` |
| `--batch-size <INT>` | Items per API batch call. | `32` |
| `--sample-size <INT>` | Randomly sample N items instead of the full dataset. | full dataset |
| `--temperature <FLOAT>` | Sampling temperature forwarded to the model. | model default |
| `--no-think` | Append `/no_think` to prompts (disables extended reasoning on supported models). | off |
| `--model-params <JSON>` | JSON string of extra parameters forwarded to the model via LiteLLM (e.g. `stop`, `max_tokens`). | none |
| `--api-key <KEY>` | API key. Overrides `API_KEY` from `.env`. | from `.env` |
| `--api-base <URL>` | Base URL for the API endpoint. Overrides `API_BASE` from `.env`. | from `.env` |

### Python API

**Full suite — run all benchmarks in one call:**

```python
from pathlib import Path
from sommbench import run

run(
    model="gpt-4o",
    output=Path("results/"),
    api_key="sk-...",
    api_base="https://api.openai.com/v1",
)
```

**Single task — run one benchmark with custom options:**

```python
import json
from sommbench import run_wtqa_benchmark

results = run_wtqa_benchmark(
    model="gpt-4o",
    language=["en", "de"],
    num_passes=2,
    api_key="sk-...",
    api_base="https://api.openai.com/v1",
    batch_size=16,
)

with open("wtqa_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
```

**Granular workflow — run each task individually and compute the composite score:**

```python
from sommbench import (
    compute_sommbench_score,
    run_fwp_benchmark,
    run_wfc_benchmark,
    run_wtqa_benchmark,
)

api_key = "sk-..."
api_base = "https://api.openai.com/v1"

wtqa = run_wtqa_benchmark(model="gpt-4o", language=None, num_passes=1, api_key=api_key, api_base=api_base)
fwp  = run_fwp_benchmark(model="gpt-4o", num_passes=1, api_key=api_key, api_base=api_base)
wfc  = run_wfc_benchmark(model="gpt-4o", language=["en", "de", "it"], num_passes=1, api_key=api_key, api_base=api_base)

score = compute_sommbench_score(wtqa, fwp, wfc)
print(score)
# {'sommbench_score': 0.6821, 's_wtqa': 0.8012, 's_fwp': 0.5432, 's_wfc': 0.7019}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use SommBench in your research, please cite:

```bibtex
@misc{brach2025sommbench,
    title={SommBench: Assessing Sommelier Expertise of Language Models},
    author={William Brach and Peter Hartman and Marek Šuppa},
    year={2025},
    eprint={2603.12117},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2603.12117},
}
```
