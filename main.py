import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable

import dspy
import typer
import weave
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt.ensemble import Ensemble
from rich.console import Console

from adapt.utils import configure_lm, dynamic_import, set_seed

print = Console(stderr=True).print

load_dotenv()

set_seed(89)

weave.init(project_name="llm-adapt-dspy")

app = typer.Typer()


def get_task_module(task_name: str):
    """Dynamically import task-specific module."""
    return dynamic_import("adapt.task", task_name)


def make_optimizer(optimizer_config: dict, metric: Callable | None = None):
    cls = dynamic_import("dspy.teleprompt", optimizer_config["class"])
    kwargs = deepcopy(optimizer_config["params"])
    if optimizer_config["with_metric"]:
        kwargs["metric"] = metric
    return cls(**kwargs)


@app.command("train")
def train_main(
    task: str = typer.Option(..., help="Name of the task"),
    dataset_path: str = typer.Option(..., help="Path to the dataset"),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_split: str = typer.Option(..., help="Dataset split to use (e.g., 'train', 'validation')"),
    model: str = typer.Option(..., help="Name of the model to use"),
    temperature: float = typer.Option(..., help="Temperature parameter for the model"),
    load_from: str = typer.Option(default="UNSET", help="Path to a saved model to load"),
    optimizer_path: Path = typer.Option(..., help="Path to the optimizer config"),
    ensemble: str = typer.Option("no", help="Whether to use an ensemble of models"),
    out: Path = typer.Option(..., help="Output file for trained program"),
):
    out.parent.mkdir(parents=True, exist_ok=True)

    # Set up LLM
    configure_lm(model, temperature)

    # Get task-specific module
    task_module = get_task_module(task)

    # Load and preprocess datasets
    examples = task_module.load_examples(dataset_path, dataset_name, dataset_split)
    print(f"Loaded {len(examples)} examples")

    # Create the program
    program = task_module.make_program()
    if load_from and load_from != "UNSET":
        print(f"Loading model from {load_from}")
        program.load(load_from)

    # Train the program
    with open(optimizer_path) as f:
        optimizer_config = json.load(f)

    if optimizer_config:
        optimizer = make_optimizer(optimizer_config, task_module.evaluate_pred)
        compile_params = optimizer_config.get("compile_params", {})
        trained_program = optimizer.compile(program, trainset=examples, **compile_params)
    else:
        trained_program = program

    if ensemble == "yes":
        ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
        candidate_programs = [x[-1] for x in trained_program.candidate_programs]
        trained_program = ensemble_optimizer.compile(candidate_programs)

    # Save the trained program
    trained_program.save(out)


@app.command("evaluate")
def evaluate_main(
    task: str = typer.Option(..., help="Name of the task"),
    dataset_path: str = typer.Option(..., help="Path to the dataset"),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_split: str = typer.Option(..., help="Dataset split to use (e.g., 'train', 'validation')"),
    model: str = typer.Option(..., help="Name of the model to use"),
    temperature: float = typer.Option(..., help="Temperature parameter for the model"),
    load_from: str = typer.Option(default="UNSET", help="Path to a saved model to load"),
    out: Path = typer.Option(..., help="Output directory for generated results"),
):
    out.mkdir(parents=True, exist_ok=True)

    # Set up LLM
    configure_lm(model, temperature)

    # Get task-specific module
    task_module = get_task_module(task)

    # Load and preprocess datasets
    examples = task_module.load_examples(dataset_path, dataset_name, dataset_split)
    print(f"Loaded {len(examples)} examples")

    # Create the program
    program = task_module.make_program()
    if load_from and load_from != "UNSET":
        print(f"Loading model from {load_from}")
        program.load(load_from)

    # Evaluate the program
    evaluate_program = Evaluate(
        metric=task_module.evaluate_pred,
        devset=examples,
        num_threads=1,
        display_progress=True,
        return_outputs=True,
    )
    _, results = evaluate_program(program)

    # Save the results
    result_df = task_module.make_results_dataframe(results)
    result_df.to_json(out / "results.jsonl", orient="records", lines=True)

    # Save the scores
    scores = task_module.aggregate_scores(result_df)
    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    dvc_exp_name = os.getenv("DVC_EXP_NAME", "default")
    with weave.attributes({"dvc.experiment": dvc_exp_name}):
        app()
