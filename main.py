import importlib
import json
import os
import time
from functools import wraps
from pathlib import Path

from cmsmusic.datasets import DatasetType
import typer
from rich.progress import track

import cmsmusic as msc


def execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"Execution time of {func.__name__}: {formatted_time}")
        return result

    return wrapper


# Create the main Typer application instance
classification_app = typer.Typer(
    help="Run event selection and classification.",
)
plotter_app = typer.Typer(
    help="Plot distributions.",
)
app = typer.Typer(
    name="lepzoo",
    help="CMS MUSiC Analysis.",
    pretty_exceptions_enable=False,
)
app.add_typer(classification_app, name="classification")
app.add_typer(plotter_app, name="plotter")


@app.command()
@execution_time
def build(
    inputs: Path = Path("datasets.py"),
):
    """
    Build analysis config.
    """

    datasets = importlib.import_module(str(inputs).replace(".py", ""))

    from datasets import datasets

    with Path("parsed_datasets.json").open("w", encoding="utf-8") as f:
        json.dump(
            [u.model_dump(mode="json") for u in datasets],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Successfully Parsed and build datasets ...")


@app.command()
@execution_time
def list_processes(
    parsed_datasets_file: Path = Path("parsed_datasets.json"),
):
    """
    List parsed datasets.
    """

    with parsed_datasets_file.open("r", encoding="utf-8") as f:
        parsed_datasets = json.load(f)
    parsed_datasets: list[msc.Dataset] = [
        msc.Dataset.model_validate(obj) for obj in parsed_datasets
    ]

    print("\nData:")
    for d in parsed_datasets:
        if d.dataset_type == DatasetType.DATA:
            print(d.short_str())

    print("\nBackground:")
    for d in parsed_datasets:
        if d.dataset_type == DatasetType.BACKGROUND:
            print(d.short_str())

    print("\nSignal:")
    for d in parsed_datasets:
        if d.dataset_type == DatasetType.SIGNAL:
            print(d.short_str())

    print()


@classification_app.command()
@execution_time
def run_serial(
    process_name: str,
    year: msc.Year,
    max_files: int = -1,
    file_index: int | None = None,
    parsed_datasets_file: Path = Path("parsed_datasets.json"),
    silence_mode: bool = False,
    enable_cache: bool = False,
):
    """
    Run selection and classification.
    """
    from cmsmusic import run_classification

    with parsed_datasets_file.open("r", encoding="utf-8") as f:
        parsed_datasets: list[msc.Dataset] = json.load(f)
    parsed_datasets: list[msc.Dataset] = [
        msc.Dataset.model_validate(obj) for obj in parsed_datasets
    ]

    if enable_cache:
        Path("nanoaod_files_cache").mkdir(parents=True, exist_ok=True)

    for dataset in parsed_datasets:
        if dataset.process_name == process_name and dataset.year == year:
            assert dataset.lfns is not None
            match file_index:
                case None:
                    for i, _ in enumerate(
                        track(
                            dataset.lfns,
                            description=f"Processing {dataset.short_str()} ...",
                            total=len(dataset.lfns),
                        )
                    ):
                        if max_files <= 0 or (max_files > 0 and i + 1 <= max_files):
                            run_classification(i, dataset, silence_mode, enable_cache)
                case int():
                    run_classification(file_index, dataset, silence_mode, enable_cache)


@classification_app.command()
@execution_time
def run_parallel(
    process_name: str | None = None,
    year: msc.Year | None = None,
    max_files: int = -1,
    parsed_datasets_file: Path = Path("parsed_datasets.json"),
):
    """
    Run selection and classification.
    """

    with parsed_datasets_file.open("r", encoding="utf-8") as f:
        parsed_datasets: list[msc.Dataset] = json.load(f)
    parsed_datasets: list[msc.Dataset] = [
        msc.Dataset.model_validate(obj) for obj in parsed_datasets
    ]

    cmds: list[str] = []
    for dataset in parsed_datasets:
        if dataset.process_name == process_name or process_name is None:
            if dataset.year == year or year is None:
                assert dataset.lfns is not None
                for i, _ in enumerate(dataset.lfns):
                    if max_files <= 0 or (max_files > 0 and i + 1 <= max_files):
                        cmds.append(
                            f"music classification run-serial {dataset.process_name} {dataset.year} --file-index {i} --silence-mode"
                        )

    Path("cmds.txt").write_text("\n".join(cmds) + "\n", encoding="utf-8")

    cmd = "parallel --results parallel_outputs --bar --retries 3 --halt soon,fail=1 --joblog joblog.tsv < cmds.txt"

    os.system("rm -rf parallel_outputs")
    os.system("mkdir -p parallel_outputs")

    rc = msc.run_stream_shell(
        cmd,
        merge_stderr=True,  # show the --bar progress
        stream_mode="auto",  # auto picks "chars" when merge_stderr=True
        shell_exe="/bin/bash",  # ensure bash features if you use them
    )
    print(f"\n[exit code: {rc}]")


@plotter_app.command()
@execution_time
def plot(
    distribution_name: str,
    force: bool = typer.Option(False, help="Brute force plot."),
):
    """
    Run plotter.
    """
    pass


if __name__ == "__main__":
    app()
