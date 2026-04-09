import json
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table

from .core import (
    compute_sommbench_score,
    run_fwp_benchmark,
    run_wfc_benchmark,
    run_wtqa_benchmark,
)

console = Console()


def print_wtqa_summary(result: dict) -> None:
    agg = result["aggregated_results"]
    table = Table(title="WTQA Results")
    table.add_column("Language")
    table.add_column("Accuracy", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Tokens", justify="right")

    langs = [k for k in agg if k != "overall"]
    for lang in sorted(langs):
        r = agg[lang]
        table.add_row(
            lang,
            f"{r['accuracy'] * 100:.1f}%",
            f"{r['correct_answers']}/{r['total_questions']}",
            str(r["total_tokens"]),
        )
    if len(langs) > 1:
        o = agg["overall"]
        table.add_row(
            "overall",
            f"{o['accuracy'] * 100:.1f}%",
            f"{o['correct_answers']}/{o['total_questions']}",
            str(o["total_tokens"]),
            style="bold",
        )
    console.print(table)


def print_fwp_summary(result: dict) -> None:
    agg = result["aggregated_results"]
    table = Table(title="FWP Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("MCC", f"{agg['mcc']:.4f}", style="bold")
    table.add_row("TPR (Sensitivity)", f"{agg['tpr']:.4f}")
    table.add_row("TNR (Specificity)", f"{agg['tnr']:.4f}")
    table.add_row("F1", f"{agg['f1']:.4f}")
    table.add_row("Accuracy", f"{agg['accuracy'] * 100:.1f}%")
    table.add_row("Score", f"{agg['correct_answers']}/{agg['total_questions']}")
    table.add_row("Total Tokens", str(agg["total_tokens"]))
    table.add_row("Avg Tokens/Question", f"{agg['avg_tokens_per_question']:.1f}")
    console.print(table)


WFC_ATTRIBUTES = [
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


def print_wfc_summary(result: dict) -> None:
    def _fmt_pct(v, digits=0):
        if v is None:
            return "-"
        return f"{v * 100:.{digits}f}%"

    agg = result["aggregated_results"]
    table = Table(title="WFC Results")
    table.add_column("Language")
    for attr in WFC_ATTRIBUTES:
        table.add_column(attr, justify="right")
    table.add_column("Mean", justify="right")

    langs = [k for k in agg if k != "overall"]
    for lang in sorted(langs):
        r = agg[lang]
        table.add_row(
            lang,
            *[_fmt_pct(r.get(a)) for a in WFC_ATTRIBUTES],
            _fmt_pct(r.get("score"), digits=1),
        )
    if len(langs) > 1:
        o = agg["overall"]
        table.add_row(
            "overall",
            *[_fmt_pct(o.get(a)) for a in WFC_ATTRIBUTES],
            _fmt_pct(o.get("score"), digits=1),
            style="bold",
        )
    console.print(table)


def print_sommbench_summary(score_dict: dict) -> None:
    table = Table(title="SommBench Score")
    table.add_column("Component")
    table.add_column("Score", justify="right")
    table.add_row("WTQA", f"{score_dict['s_wtqa']:.4f}")
    table.add_row("FWP", f"{score_dict['s_fwp']:.4f}")
    table.add_row("WFC", f"{score_dict['s_wfc']:.4f}")
    table.add_row(
        "SommBench Score",
        f"{score_dict['sommbench_score']:.4f}",
        style="bold",
    )
    console.print(table)


load_dotenv()

app = typer.Typer()


supported_langs = ["all", "sk", "en", "de", "da", "fi", "sv", "it", "es"]


def resolve_output_path(output: Path | None) -> Path:
    if output is None:
        print("[bold red]Error: No output file path provided.[/bold red]")
        raise typer.Abort()

    if output.is_dir():
        print(f"➡️ Output is a directory. Results will be saved inside: {output}")
        output /= "results.json"

    elif not output.exists() and not output.suffix:
        print(f"➡️ Path has no extension. Assuming it's a directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        output /= "results.json"

    else:  # Path is a file or doesn't exist but has a suffix
        if output.exists():
            print(f"⚠️ Output file exists and will be overwritten: {output}")
        else:
            print(
                f"✅ Output path does not exist. Creating necessary directories for: {output}"
            )

        # Ensure parent directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Final output file will be: {output}")
    return output


def validate_languages(language: list[str]) -> None:
    languages_to_check: set[str] = set(language)
    supported_langs_set: set[str] = set(supported_langs)

    unsupported = languages_to_check - supported_langs_set

    if unsupported:
        unsupported_str = ", ".join(f"'{lang}'" for lang in sorted(unsupported))
        plural_s = "s" if len(unsupported) > 1 else ""
        verb = "are" if len(unsupported) > 1 else "is"
        print(
            f"[bold red]Language{plural_s} {unsupported_str} {verb} not supported. "
            f"Supported languages are: {', '.join(supported_langs)}[/bold red]"
        )
        raise typer.Exit(code=1)


@app.command()
def wtqa(
    model: Annotated[str, typer.Argument(help="The name of the model to test")],
    output: Annotated[
        Path | None,
        typer.Option(
            help="The output file to save results to",
            exists=False,
            file_okay=True,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("results.json"),
    language: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            "-l",
            help=f"Select languages of the benchmark. Can be used multiple times. Use 'all' for all. Supported languages: {', '.join(supported_langs)}",
        ),
    ] = None,
    num_passes: Annotated[
        int, typer.Option(help="Number of full benchmark passes")
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="Batch size for API calls")] = 32,
    api_key: Annotated[
        str | None, typer.Option(help="The API key for the model", envvar="API_KEY")
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(help="The API base URL for the model", envvar="API_BASE"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to use for each benchmark (if applicable).",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(help="Sampling temperature for the model."),
    ] = None,
    no_think: Annotated[
        bool,
        typer.Option(
            "--no-think",
            help="Append /no_think to prompts (disables reasoning for supported models)",
        ),
    ] = False,
    model_params: Annotated[
        str | None,
        typer.Option(
            "--model-params",
            help='JSON string of extra model parameters (e.g. \'{"stop": ["<|im_end|>"], "max_tokens": 512}\')',
        ),
    ] = None,
) -> dict:
    """
    Wine Theory Question-Answering benchmark
    """
    if language is None:
        language = ["all"]
    validate_languages(language)
    if api_key is None or api_base is None:
        print("[bold red]Error: API key and API base must be provided.[/bold red]")
        raise typer.Exit(code=1)
    output = resolve_output_path(output)

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model_params is not None:
        kwargs.update(json.loads(model_params))

    result = run_wtqa_benchmark(
        model=model,
        language=None if "all" in language else language,
        num_passes=num_passes,
        api_key=api_key,
        api_base=api_base,
        batch_size=batch_size,
        sample_size=sample_size,
        no_think=no_think,
        **kwargs,
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print_wtqa_summary(result)
    print(f"Results saved to {output}")
    return result


@app.command()
def fwp(
    model: Annotated[str, typer.Argument(help="The name of the model to test")],
    output: Annotated[
        Path | None,
        typer.Option(
            help="The output file to save results to",
            exists=False,
            file_okay=True,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("results.json"),
    num_passes: Annotated[
        int, typer.Option(help="Number of full benchmark passes")
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="Batch size for API calls")] = 32,
    api_key: Annotated[
        str | None, typer.Option(help="The API key for the model", envvar="API_KEY")
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(help="The API base URL for the model", envvar="API_BASE"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to use for each benchmark (if applicable).",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(help="Sampling temperature for the model."),
    ] = None,
    no_think: Annotated[
        bool,
        typer.Option(
            "--no-think",
            help="Append /no_think to prompts (disables reasoning for supported models)",
        ),
    ] = False,
    model_params: Annotated[
        str | None,
        typer.Option(
            "--model-params",
            help='JSON string of extra model parameters (e.g. \'{"stop": ["<|im_end|>"], "max_tokens": 512}\')',
        ),
    ] = None,
) -> dict:
    """
    Food & Wine Pairing benchmark
    """
    if api_key is None or api_base is None:
        print("[bold red]Error: API key and API base must be provided.[/bold red]")
        raise typer.Exit(code=1)
    output = resolve_output_path(output)

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model_params is not None:
        kwargs.update(json.loads(model_params))

    result = run_fwp_benchmark(
        model=model,
        num_passes=num_passes,
        api_key=api_key,
        api_base=api_base,
        batch_size=batch_size,
        sample_size=sample_size,
        no_think=no_think,
        **kwargs,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print_fwp_summary(result)
    print(f"Results saved to {output}")
    return result


@app.command()
def wfc(
    model: Annotated[str, typer.Argument(help="The name of the model to test")],
    output: Annotated[
        Path | None,
        typer.Option(
            help="The output file to save results to",
            exists=False,
            file_okay=True,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("results.json"),
    language: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            "-l",
            help=f"Select languages of the benchmark. Can be used multiple times. Use 'all' for all. Supported languages: {', '.join(supported_langs)}",
        ),
    ] = None,
    num_passes: Annotated[
        int, typer.Option(help="Number of full benchmark passes")
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="Batch size for API calls")] = 32,
    api_key: Annotated[
        str | None, typer.Option(help="The API key for the model", envvar="API_KEY")
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(help="The API base URL for the model", envvar="API_BASE"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to use for each benchmark (if applicable).",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(help="Sampling temperature for the model."),
    ] = None,
    no_think: Annotated[
        bool,
        typer.Option(
            "--no-think",
            help="Append /no_think to prompts (disables reasoning for supported models)",
        ),
    ] = False,
    model_params: Annotated[
        str | None,
        typer.Option(
            "--model-params",
            help='JSON string of extra model parameters (e.g. \'{"stop": ["<|im_end|>"], "max_tokens": 512}\')',
        ),
    ] = None,
) -> dict:
    """
    Wine Features Generation benchmark
    """
    if language is None:
        language = ["sk", "en", "de", "da", "fi", "sv", "it", "es"]
    validate_languages(language)
    if api_key is None or api_base is None:
        print("[bold red]Error: API key and API base must be provided.[/bold red]")
        raise typer.Exit(code=1)
    output = resolve_output_path(output)

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model_params is not None:
        kwargs.update(json.loads(model_params))

    result = run_wfc_benchmark(
        model=model,
        language=language,
        num_passes=num_passes,
        api_key=api_key,
        api_base=api_base,
        batch_size=batch_size,
        sample_size=sample_size,
        no_think=no_think,
        **kwargs,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print_wfc_summary(result)
    print(f"Results saved to {output}")
    return result


@app.command()
def run(
    model: Annotated[str, typer.Argument(help="The name of the model to test")],
    output: Annotated[
        Path | None,
        typer.Option(
            help="The output directory to save result files to.",
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("results/"),
    language: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            "-l",
            help=f"Select languages for all benchmarks. Can be used multiple times. Use 'all' for all. Supported languages: {', '.join(supported_langs)}",
        ),
    ] = None,
    num_passes: Annotated[
        int, typer.Option(help="Number of full benchmark passes")
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="Batch size for API calls")] = 32,
    api_key: Annotated[
        str | None, typer.Option(help="The API key for the model", envvar="API_KEY")
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(help="The API base URL for the model", envvar="API_BASE"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to use for each benchmark (if applicable).",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(help="Sampling temperature for the model."),
    ] = None,
    no_think: Annotated[
        bool,
        typer.Option(
            "--no-think",
            help="Append /no_think to prompts (disables reasoning for supported models)",
        ),
    ] = False,
    model_params: Annotated[
        str | None,
        typer.Option(
            "--model-params",
            help='JSON string of extra model parameters (e.g. \'{"stop": ["<|im_end|>"], "max_tokens": 512}\')',
        ),
    ] = None,
) -> None:
    """
    Run all benchmarks (wtqa, fwp, wfc) and save each result to a separate file.
    """
    if language is None:
        language = ["all"]
    validate_languages(language)
    if api_key is None or api_base is None:
        print("[bold red]Error: API key and API base must be provided.[/bold red]")
        raise typer.Exit(code=1)

    if output is None:
        print("[bold red]Error: No output directory path provided.[/bold red]")
        raise typer.Abort()

    # Ensure the output directory exists
    output.mkdir(parents=True, exist_ok=True)
    print(f"✅ Results will be saved in directory: {output}")

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model_params is not None:
        kwargs.update(json.loads(model_params))

    results: dict[str, dict] = {}

    for name in ["wtqa", "fwp", "wfc"]:
        lang_arg: list[str] | None = language
        if name == "wtqa":
            lang_arg = None if "all" in language else language
        elif name == "wfc" and ("all" in language or not language):
            lang_arg = [lang for lang in supported_langs if lang != "all"]

        model_name = model.replace("/", "_")
        output_file = output / f"{model_name}_{name}.json"

        print(f"\n[bold cyan]Running benchmark: {name}...[/bold cyan]")
        print(
            f"With parameters: model={model_name}, language={lang_arg}, num_passes={num_passes}, batch_size={batch_size}"
        )

        if name == "wtqa":
            result = run_wtqa_benchmark(
                model=model,
                language=lang_arg,
                num_passes=num_passes,
                api_key=api_key,
                api_base=api_base,
                batch_size=batch_size,
                sample_size=sample_size,
                no_think=no_think,
                **kwargs,
            )
        elif name == "fwp":
            result = run_fwp_benchmark(
                model=model,
                num_passes=num_passes,
                api_key=api_key,
                api_base=api_base,
                batch_size=batch_size,
                sample_size=sample_size,
                no_think=no_think,
                **kwargs,
            )
        else:  # wfc
            assert isinstance(lang_arg, list)
            result = run_wfc_benchmark(
                model=model,
                language=lang_arg,
                num_passes=num_passes,
                api_key=api_key,
                api_base=api_base,
                batch_size=batch_size,
                sample_size=sample_size,
                no_think=no_think,
                **kwargs,
            )

        results[name] = result

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        if name == "wtqa":
            print_wtqa_summary(result)
        elif name == "fwp":
            print_fwp_summary(result)
        else:
            print_wfc_summary(result)

        print(
            f"[bold green]Finished '{name}'. Results saved to {output_file}[/bold green]"
        )

    # Compute and display SommBench Score
    score_dict = compute_sommbench_score(
        results["wtqa"], results["fwp"], results["wfc"]
    )
    print_sommbench_summary(score_dict)

    model_name = model.replace("/", "_")
    score_file = output / f"{model_name}_sommbench_score.json"
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, indent=4, ensure_ascii=False)
    print(f"[bold green]SommBench Score saved to {score_file}[/bold green]")

    print("\n[bold magenta]All benchmarks completed successfully![/bold magenta]")


if __name__ == "__main__":
    app()
