import contextlib  # noqa: I001
import json
import re
import time
from typing import Any

import litellm
import numpy as np
import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel
from rich import print
from rich.progress import track
from sklearn.metrics import matthews_corrcoef, mean_absolute_percentage_error

from .data_utils import build_wine_passage, create_tiered_mask_dataset
from .prompts import (
    build_fwp_prompt,
    build_wfc_prompt,
    build_wtqa_prompt,
    get_wfc_schema,
)
from .translations import COUNTRY_TRANSLATIONS_EN_MAP, TRANSLATION_MAP

_HF_REPO_ID = "sommify/sommbench"


def _load_benchmark_data(config: str) -> pd.DataFrame:
    """Load benchmark data from HuggingFace Hub, falling back to local CSV."""
    try:

        ds = load_dataset(_HF_REPO_ID, config, split="test")
        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame for config '{config}', got {type(df)}"
        assert not df.empty, f"Loaded empty dataset for config '{config}' from HuggingFace Hub"

        if config == "wfc":
            for col in df.columns:
                if col in ("region", "grapes"):
                    continue
                if df[col].dtype != object:
                    continue
                df[col] = df[col].replace("", float("nan"))
        return df
    except Exception as e:
        raise ValueError(f"Error loading {config} data: {e}")



def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a litellm response, defaulting to 0."""
    try:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except Exception as e:
        print(f"Error extracting usage: {e}")
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def compute_binary_classification_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, float]:
    """Compute MCC, TPR, TNR, F1, and accuracy for binary yes/no classification.

    ``"yes"`` is the positive class; anything else (including ``"error"``)
    is treated as negative.
    """
    pos = "yes"
    t = [1 if v == pos else 0 for v in y_true]
    p = [1 if v == pos else 0 for v in y_pred]

    tp = sum(ti == 1 and pi == 1 for ti, pi in zip(t, p, strict=True))
    tn = sum(ti == 0 and pi == 0 for ti, pi in zip(t, p, strict=True))
    fp = sum(ti == 0 and pi == 1 for ti, pi in zip(t, p, strict=True))
    fn = sum(ti == 1 and pi == 0 for ti, pi in zip(t, p, strict=True))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
        if (precision + recall) > 0
        else 0.0
    )
    mcc = float(matthews_corrcoef(t, p))

    return {
        "mcc": round(mcc, 4),
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _run_batch_completion(
    messages_list: list[list[dict]],
    model: str,
    api_key: str,
    api_base: str,
    batch_size: int,
    **kwargs,
) -> list:
    """Run litellm completion in batches, returning a flat list of responses."""
    batches = [
        messages_list[i : i + batch_size]
        for i in range(0, len(messages_list), batch_size)
    ]
    responses: list = []
    for batch in track(batches, description="Processing batches..."):
        try:
            if batch_size == 1:
                response = litellm.completion(
                    model=model,
                    messages=batch[0],
                    api_key=api_key,
                    api_base=api_base,
                    **kwargs,
                )
                responses.append(response)
            else:
                response = litellm.batch_completion(
                    model=model,
                    messages=batch,
                    api_key=api_key,
                    api_base=api_base,
                    **kwargs,
                )
                responses.extend(response)
        except Exception as e:
            print(f"Error during batch processing: {e}")
            if batch_size == 1:
                responses.append(None)
            else:
                responses.extend([None] * len(batch))
        time.sleep(1)
    return responses


def clean_wtqa_answer(answer: str, lang: str) -> str:
    # Strip common special/chat-template tokens (e.g. <|im_end|>, <|endoftext|>)
    answer = re.sub(r"<\|.*?\|>", "", answer).strip()
    if lang == "it":
        answer = (
            answer.replace("la risposta è ", "")
            .replace("la risposta corretta è", "")
            .replace("la risposta è", "")
            .strip()
            .strip(".")
            .strip()
        )
    if lang == "es":
        answer = (
            answer.replace("la respuesta es:", "")
            .replace("la respuesta correcta es", "")
            .replace("la respuesta es", "")
            .strip()
            .strip(".")
            .strip()
        )
    if lang == "fi":
        answer = (
            answer.replace("oikea vastaus on", "")
            .replace("vastaukset:", "")
            .replace("vastauksen oikea vastaus on", "")
            .strip()
            .strip(".")
            .strip()
        )
    if lang == "de":
        answer = (
            answer.replace("die antwort ist", "")
            .replace("die richtige antwort ist:", "")
            .replace("die richtige antwort ist ", "")
            .replace("die richtige antwort lautet:", "")
            .replace("antwort:", "")
            .strip()
            .strip(".")
            .strip()
        )
    if "(a)" in answer:
        answer = "a"
    elif "(b)" in answer:
        answer = "b"
    elif "(c)" in answer:
        answer = "c"
    elif "(d)" in answer:
        answer = "d"
    else:
        answer = answer
    return answer


def _aggregate_wtqa_overall(
    aggregated_results: dict[str, dict[str, float | int]],
    languages: list[str],
) -> dict[str, float | int]:
    overall_results: dict[str, list[float | int]] = {
        "accuracy": [],
        "total_questions": [],
        "correct_answers": [],
        "total_tokens": [],
        "avg_tokens_per_question": [],
        "total_prompt_tokens": [],
        "total_completion_tokens": [],
    }
    for lang in languages:
        lang_results = aggregated_results[lang]
        assert isinstance(lang_results, dict)
        overall_results["accuracy"].append(lang_results["accuracy"])
        overall_results["total_questions"].append(lang_results["total_questions"])
        overall_results["correct_answers"].append(lang_results["correct_answers"])
        overall_results["total_tokens"].append(lang_results["total_tokens"])
        overall_results["avg_tokens_per_question"].append(
            lang_results["avg_tokens_per_question"]
        )
        overall_results["total_prompt_tokens"].append(
            lang_results["total_prompt_tokens"]
        )
        overall_results["total_completion_tokens"].append(
            lang_results["total_completion_tokens"]
        )

    return {
        "accuracy": round(float(np.mean(overall_results["accuracy"])), 3),
        "total_questions": int(np.sum(overall_results["total_questions"])),
        "correct_answers": int(np.sum(overall_results["correct_answers"])),
        "total_tokens": int(np.sum(overall_results["total_tokens"])),
        "avg_tokens_per_question": round(
            float(np.mean(overall_results["avg_tokens_per_question"])), 3
        ),
        "total_prompt_tokens": int(np.sum(overall_results["total_prompt_tokens"])),
        "total_completion_tokens": int(
            np.sum(overall_results["total_completion_tokens"])
        ),
    }


def run_wtqa_benchmark(
    model: str,
    language: list[str] | str | None,
    num_passes: int,
    api_key: str,
    api_base: str,
    batch_size: int = 32,
    sample_size: int | None = None,
    no_think: bool = False,
    **kwargs,
) -> dict:

    aggregated_results: dict[str, dict[str, float | int]] = {}
    wtqa_instances: dict[str, list[Any]] = {}
    output = {
        "run_metadata": {
            "model": model,
            "language": language if language is not None else "all",
            "num_passes": num_passes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "parameters": kwargs,
        },
        "aggregated_results": aggregated_results,
        "instances": wtqa_instances,
    }

    df = _load_benchmark_data("wtqa")
    if language is not None and language != "all":
        if isinstance(language, str):
            df = df[df["language"] == language]
        elif isinstance(language, list):
            df = df[df["language"].isin(language)]
        else:
            raise ValueError("language must be a string or a list of strings")

    print(
        f"Running WTQA benchmark on {len(df)} questions and with language: {','.join(df['language'].unique())}"
    )
    for lang in df["language"].unique():
        print(f"Running benchmark for language: {lang}")
        lang_df = df[df["language"] == lang]
        if sample_size is not None:
            lang_df = lang_df.sample(sample_size, random_state=42).reset_index(
                drop=True
            )
            print(f"Sampled [{lang}] down to {sample_size} instances for benchmarking.")
        true_answers = [ans.lower().strip() for ans in lang_df["true_label"].tolist()]
        messages_list = []
        for _, row in lang_df.iterrows():
            messages_list.append(
                [
                    {
                        "content": build_wtqa_prompt(
                            question=row["question"],
                            a=row["a"],
                            b=row["b"],
                            c=row["c"],
                            d=row["d"],
                            lang=lang,
                        ),
                        "role": "user",
                    }
                ]
            )
        if no_think:
            for query_messages in messages_list:
                if query_messages:
                    query_messages[-1]["content"] += " /no_think"

        language_instances = []
        for _ in track(range(num_passes), description="Processing..."):
            iteration_instances = []
            responses = _run_batch_completion(
                messages_list, model, api_key, api_base, batch_size, **kwargs
            )
            for response, true_answer, prompt in zip(
                responses, true_answers, messages_list, strict=True
            ):
                usage = _extract_usage(response)
                try:
                    answer = (
                        response.choices[0].message.content.lower().strip().strip(".")
                    )
                    answer = clean_wtqa_answer(answer, lang)
                    is_correct = answer == true_answer
                    iteration_instances.append(
                        {
                            "prompt": prompt[0]["content"],
                            "ground_truth": true_answer,
                            "llm_output": answer,
                            "is_correct": is_correct,
                            "usage": usage,
                        }
                    )
                except Exception as e:
                    print(f"Error processing response: {e}")
                    iteration_instances.append(
                        {
                            "prompt": prompt[0]["content"],
                            "ground_truth": true_answer,
                            "llm_output": None,
                            "is_correct": False,
                            "usage": usage,
                        }
                    )
            language_instances.append(iteration_instances)
        wtqa_instances[lang] = language_instances
        flatterned_language_instances = [
            instance for sublist in language_instances for instance in sublist
        ]
        aggregated_results[lang] = {
            "accuracy": round(
                sum(inst["is_correct"] for inst in flatterned_language_instances)
                / len(flatterned_language_instances),
                3,
            ),
            "total_questions": len(flatterned_language_instances),
            "correct_answers": sum(
                inst["is_correct"] for inst in flatterned_language_instances
            ),
            "total_tokens": sum(
                inst["usage"]["total_tokens"] for inst in flatterned_language_instances
            ),
            "avg_tokens_per_question": round(
                sum(
                    inst["usage"]["total_tokens"]
                    for inst in flatterned_language_instances
                )
                / len(flatterned_language_instances),
                3,
            ),
            "total_prompt_tokens": sum(
                inst["usage"]["prompt_tokens"] for inst in flatterned_language_instances
            ),
            "total_completion_tokens": sum(
                inst["usage"]["completion_tokens"]
                for inst in flatterned_language_instances
            ),
        }

    aggregated_results["overall"] = _aggregate_wtqa_overall(
        aggregated_results, list(df["language"].unique())
    )

    return output


def run_fwp_benchmark(
    model: str,
    num_passes: int,
    api_key: str,
    api_base: str,
    batch_size: int = 32,
    sample_size: int | None = None,
    no_think: bool = False,
    **kwargs,
) -> dict:

    instances: list[list[dict[str, Any]]] = []
    output = {
        "run_metadata": {
            "model": model,
            "num_passes": num_passes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "parameters": kwargs,
        },
        "aggregated_results": {},
        "instances": instances,
    }

    df = _load_benchmark_data("fwp")
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {sample_size} instances for benchmarking.")

    true_answers = [ans.lower().strip(".").strip() for ans in df["true_label"].tolist()]
    # build messages
    messages_list = []
    for _, row in df.iterrows():
        messages_list.append(
            [
                {
                    "content": build_fwp_prompt(
                        recipe=row["recipe"],
                        wine=row["wine"],
                    ),
                    "role": "user",
                }
            ]
        )
    if no_think:
        for query_messages in messages_list:
            if query_messages:
                query_messages[-1]["content"] += " /no_think"
    print(f"Running FWP benchmark on {len(messages_list)} questions.")

    for _ in track(range(num_passes), description="Processing..."):
        iteration_instances = []
        responses = _run_batch_completion(
            messages_list, model, api_key, api_base, batch_size, **kwargs
        )
        for response, true_answer, prompt in zip(
            responses, true_answers, messages_list, strict=True
        ):
            try:
                answer = response.choices[0].message.content.lower().strip(".").strip()
            except Exception as e:
                print(f"Error processing response: {e} - setting answer to 'error'")
                print(f"Response object: {response}")
                answer = "error"
            is_correct = answer == true_answer
            usage = _extract_usage(response)
            iteration_instances.append(
                {
                    "prompt": prompt[0]["content"],
                    "ground_truth": true_answer,
                    "llm_output": answer,
                    "is_correct": is_correct,
                    "usage": usage,
                }
            )
        instances.append(iteration_instances)

    flatterned_instances = [
        instance for sublist in instances for instance in sublist
    ]

    y_true = [inst["ground_truth"] for inst in flatterned_instances]
    y_pred = [inst["llm_output"] for inst in flatterned_instances]
    clf_metrics = compute_binary_classification_metrics(y_true, y_pred)

    output["aggregated_results"] = {
        **clf_metrics,
        "total_questions": len(flatterned_instances),
        "correct_answers": sum(inst["is_correct"] for inst in flatterned_instances),
        "total_tokens": sum(
            inst["usage"]["total_tokens"] for inst in flatterned_instances
        ),
        "avg_tokens_per_question": round(
            sum(inst["usage"]["total_tokens"] for inst in flatterned_instances)
            / len(flatterned_instances),
            3,
        ),
        "total_prompt_tokens": sum(
            inst["usage"]["prompt_tokens"] for inst in flatterned_instances
        ),
        "total_completion_tokens": sum(
            inst["usage"]["completion_tokens"] for inst in flatterned_instances
        ),
    }

    return output


def run_wfc_benchmark(
    model: str,
    language: list[str],
    num_passes: int,
    api_key: str,
    api_base: str,
    batch_size: int = 32,
    mask_token: str = "[MASK]",
    sample_size: int | None = None,
    no_think: bool = False,
    **kwargs,
) -> dict:

    wfc_aggregated: dict[str, dict[str, Any]] = {}
    wfc_instances: dict[str, list[Any]] = {}
    output = {
        "run_metadata": {
            "model": model,
            "language": language if language is not None else "all",
            "num_passes": num_passes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "parameters": kwargs,
        },
        "aggregated_results": wfc_aggregated,
        "instances": wfc_instances,
    }
    df = _load_benchmark_data("wfc")

    df = create_tiered_mask_dataset(df, mask_token=mask_token, seed=42)
    df["wine_passage"] = df.apply(build_wine_passage, axis=1)

    if sample_size is not None:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {sample_size} instances for benchmarking.")
    for lang in language:
        print(f"Running benchmark for language: {lang}")
        language_instances = []
        for _ in track(range(num_passes), description="Processing..."):
            iteration_instances = []
            schema = get_wfc_schema(lang)
            messages_list = []
            for _, row in df.iterrows():
                messages_list.append(
                    [
                        {
                            "content": build_wfc_prompt(
                                passage=row["wine_passage"],
                                language=lang,
                            ),
                            "role": "user",
                        },
                    ]
                )
            if no_think:
                for query_messages in messages_list:
                    if query_messages:
                        query_messages[-1]["content"] += " /no_think"

            responses = _run_batch_completion(
                messages_list,
                model,
                api_key,
                api_base,
                batch_size,
                response_format=schema,
                **kwargs,
            )
            for response, (_, row), prompt in zip(
                responses, df.iterrows(), messages_list, strict=True
            ):
                answer = get_dict_from_response(response, schema)

                true = (
                    row[
                        [
                            "true_type",
                            "true_sugar",
                            "true_alcohol",
                            "true_country",
                            "true_region",
                            "true_grapes",
                            "true_dryness",
                            "true_body",
                            "true_acidity",
                        ]
                    ]
                    .rename(lambda x: x.replace("true_", ""))
                    .to_dict()
                )
                # if nan in true set to None; convert ndarrays to lists
                for k, v in true.items():
                    if isinstance(v, float) and np.isnan(v):
                        true[k] = None
                    elif isinstance(v, np.ndarray):
                        true[k] = v.tolist()

                masked_attributes = [
                    attr
                    for attr in [
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
                    if isinstance(row[attr], str) and row[attr] == mask_token
                ]
                usage = _extract_usage(response)

                iteration_instances.append(
                    {
                        "prompt": prompt[0]["content"],
                        "ground_truth": true,
                        "llm_output": answer,
                        "is_correct": score_wfc_prediction(
                            answer, true, masked_attributes
                        ),
                        # "schema": schema.model_json_schema(),
                        "language": lang,
                        "masked_attributes": masked_attributes,
                        "usage": usage,
                    }
                )
            language_instances.append(iteration_instances)
        wfc_instances[lang] = language_instances
    for lang in wfc_instances:
        lang_instance_lists = wfc_instances[lang]
        language_score = {}
        flat_instances = [item for sublist in lang_instance_lists for item in sublist]
        scores = [inst["is_correct"] for inst in flat_instances]
        score_df = pd.DataFrame(scores)
        attr_means = score_df[
            [
                "type",
                "country",
                "dryness",
                "body",
                "acidity",
                "region",
                "grapes",
                "sugar",
                "alcohol",
            ]
        ].mean()
        language_score = {k: (None if pd.isna(v) else v) for k, v in attr_means.to_dict().items()}
        mean_val = attr_means.mean()
        language_score["score"] = None if pd.isna(mean_val) else round(float(mean_val), 4)
        wfc_aggregated[lang] = language_score

    overall_score = {}
    for lang in wfc_aggregated:
        lang_results = wfc_aggregated[lang]
        for key in lang_results:
            if key == lang:
                continue
            if key not in overall_score:
                overall_score[key] = []
            overall_score[key].append(lang_results[key])
    overall_score = {
        key: (round(float(sum(valid) / len(valid)), 2) if (valid := [v for v in value if v is not None]) else None)
        for key, value in overall_score.items()
    }
    # S_WFC = mean of per-language scores
    lang_scores = [
        wfc_aggregated[lang]["score"]
        for lang in wfc_aggregated
    ]
    valid_lang_scores = [s for s in lang_scores if s is not None]
    overall_score["score"] = round(float(np.mean(valid_lang_scores)), 4) if valid_lang_scores else None
    wfc_aggregated["overall"] = overall_score
    return output


def calculate_mape(true: int | float | list, pred: int | float | list) -> float:
    if isinstance(true, int | float):
        true = [true]
    if isinstance(pred, int | float):
        pred = [pred]
    return mean_absolute_percentage_error(true, pred) * 100


def score_wfc_prediction(pred: dict, true: dict, masked_attributes: list) -> dict:
    output = {}
    # handle type

    # string category types
    for column in ["type", "country", "dryness", "body", "acidity"]:
        x = str(pred[column]).lower().strip()
        if column in TRANSLATION_MAP:
            if x in TRANSLATION_MAP[column]:
                x = TRANSLATION_MAP[column][x]
            else:
                if x is not None and x != "none":
                    print(f"Could not find translation for predicted {column}: {x}")

        if column == "country" and x is not None:
            x = COUNTRY_TRANSLATIONS_EN_MAP.get(str(x).lower(), str(x).lower()).lower()

        y = str(true[column]).lower().strip()
        if y == "none" or y == "nan":
            output[column] = 0
            continue
        if column == "type":
            output[column] = int(x in y)
        else:
            output[column] = int(x == y)
    # float numeric types
    for column in ["sugar", "alcohol"]:
        try:
            x = float(pred[column])
            y = float(true[column])
            output[f"{column}_mape"] = calculate_mape(y, x)
            mape_threshhold = 5
            output[column] = 1 if output[f"{column}_mape"] < mape_threshhold else 0
        except Exception:
            print(
                f"Could not convert {column} to float: pred={pred[column]}, true={true[column]}"
            )
            output[column] = 0.0
    try:
        raw = pred["region"]
        if isinstance(raw, list):
            pred_regions = [r.lower().strip() for r in raw]
        else:
            pred_regions = [r.lower().strip() for r in str(raw).split(",")]
        true_regions = [r.lower().strip() for r in true["region"]]
        output["region"] = int(any(pr in true_regions for pr in pred_regions))
    except Exception:
        print(f"Could not process region: pred={pred['region']}, true={true['region']}")
        output["region"] = 0

    # handle grapes if one of pred grapes is in true grapes 1 else 0
    try:
        pred_grapes = [g.lower().strip() for g in pred["grapes"]]
        true_grapes = [g.lower().strip() for g in true["grapes"]]
        output["grapes"] = int(any(g in true_grapes for g in pred_grapes))
    except Exception:
        print(f"Could not process grapes: pred={pred['grapes']}, true={true['grapes']}")
        output["grapes"] = 0

    for attr in list(output.keys()):
        base_attr = attr.replace("_mape", "")
        if base_attr not in masked_attributes:
            output[attr] = None
    return output


def get_dict_from_response(response: Any, schema: type[BaseModel]) -> dict:
    try:
        if response is None:
            return {
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
        extracted_data = response.choices[0].message.content
        if extracted_data is None:
            return {
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
        if isinstance(response.choices[0].message.content, schema):
            extracted_data = extracted_data.model_dump()
        else:
            extracted_data = json.loads(extracted_data)

        for key in [
            "type",
            "sugar",
            "alcohol",
            "country",
            "region",
            "grapes",
            "dryness",
            "body",
            "acidity",
        ]:
            with contextlib.suppress(Exception):
                extracted_data[key] = (
                    extracted_data[key].encode("latin-1").decode("utf-8")
                )
                # how to handle lists of strings with encoding issues?
                if isinstance(extracted_data[key], list):
                    extracted_data[key] = [
                        g.encode("latin-1").decode("utf-8") for g in extracted_data[key]
                    ]
        for key in [
            "type",
            "sugar",
            "alcohol",
            "country",
            "region",
            "grapes",
            "dryness",
            "body",
            "acidity",
        ]:
            if key not in extracted_data:
                extracted_data[key] = None
        # Normalize empty values to None
        for key in ["type", "country", "region", "dryness", "body", "acidity"]:
            if (
                isinstance(extracted_data.get(key), str)
                and not extracted_data[key].strip()
            ):
                extracted_data[key] = None
        for key in ["grapes"]:
            if (
                isinstance(extracted_data.get(key), list)
                and len(extracted_data[key]) == 0
            ):
                extracted_data[key] = None
        return extracted_data
    except Exception as e:
        print(f"Error extracting data from response: {e}")
        return {
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


def compute_sommbench_score(
    wtqa_result: dict, fwp_result: dict, wfc_result: dict
) -> dict[str, float]:
    """Compute the composite SommBench Score from the three benchmark results.

    SommBench-Score = (S_FWP + S_WTQA + S_WFC) / 3
    """
    s_wtqa = wtqa_result["aggregated_results"]["overall"]["accuracy"]
    s_fwp = fwp_result["aggregated_results"]["mcc"]
    s_wfc = wfc_result["aggregated_results"]["overall"]["score"]

    return {
        "sommbench_score": round((s_fwp + s_wtqa + s_wfc) / 3, 4),
        "s_wtqa": round(s_wtqa, 4),
        "s_fwp": round(s_fwp, 4),
        "s_wfc": round(s_wfc, 4),
    }
