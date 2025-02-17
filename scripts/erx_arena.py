import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

tqdm.pandas()

load_dotenv()

app = typer.Typer()


def set_seed(seed):
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


SEED = 89
set_seed(SEED)


def jprint(obj):
    print(json.dumps(obj, indent=2))


client = OpenAI(max_retries=3)


def silent(exception_class=Exception):
    """
    A decorator to silence errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_class:
                return None

        return wrapper

    return decorator


class Result(BaseModel):
    explanation: str
    decision: str = Field(description="A or B or DRAW")


SCHEMA = Result.model_json_schema()

SYSTEM_PROMPT = """
Evaluate and compare two sets of extracted triples (`subject | relation | object`) from the given text. Each set was produced by a different joint entity relation extraction model. Assess them based on the following ranked criteria:

1. **Completeness** – Does the set capture all explicitly or unambiguously stated relationships?  
2. **Correctness** – Are all triples factually accurate, with correct entities, relations, and directionality?  
3. **Consistency with Text** – Do the triples strictly align with the text without adding assumptions?  
4. **Precision vs. Overgeneration** – Does the set avoid extracting unnecessary, vague, or incorrect triples?  
5. **No Redundancy** – Are there no duplicate or near-duplicate triples?  
6. **Informative Representation** – Are the triples concise yet meaningful, capturing the right level of detail?  

### Evaluation Steps:  
- **Completeness Check**: Identify any missing key relationships.  
- **Error Detection**: Spot incorrect entities, relations, or order.  
- **Faithfulness**: Ensure no hallucinated or misinterpreted triples.  
- **Conciseness**: Highlight redundancy or excessive generalization.  
- **Final Verdict**: Compare both sets and determine which is better.  
  - If one set is significantly better, declare it the winner.  
  - If both sets perform similarly with only trivial differences, **declare a DRAW** instead of forcing a winner.  

# Text:  
{text}  

# Model A Triples:  
{triples_a}  

# Model B Triples:  
{triples_b}  

Output your structured comparison in the following JSON format:  
{schema}  
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3), reraise=True)
def compare_triples_with_llm(text, triples_a, triples_b, model: str, temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(text=text, triples_a=triples_a, triples_b=triples_b, schema=SCHEMA),
            }
        ],
        response_format={"type": "json_object", "value": SCHEMA},
    )
    return Result.model_validate_json(response.choices[0].message.content)


@silent(ValidationError)
def compare_triples(text, triples_a, triples_b, model: str, temperature=0.3, flip=False):
    # randomize the order of triples in the prompt between A and B
    if flip:
        triples_a, triples_b = triples_b, triples_a
        mapping = {"A": "B", "B": "A", "DRAW": "DRAW"}
    else:
        mapping = {"A": "A", "B": "B", "DRAW": "DRAW"}

    result = compare_triples_with_llm(text, triples_a, triples_b, model=model, temperature=temperature)
    result.decision = mapping.get(result.decision)
    return {**result.model_dump(), "flipped": flip}


def process(row, model: str, temperature=0.3):
    result = compare_triples(
        row["text"],
        row["predicted_triples_A"],
        row["predicted_triples_B"],
        model=model,
        temperature=temperature,
        flip=random.random() < 0.5,
    )
    return {**row, **(result or {})}


def process_dataframe(dataf, model: str, temperature=0.3, num_threads=16):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, row, model, temperature) for _, row in dataf.iterrows()]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing row: {e}")
                results.append(None)
    return pd.DataFrame(results)


def compute_stats(df):
    return df["decision"].value_counts().to_dict()


def compare_pair(
    file_a,
    file_b,
    output_dir: Path,
    model: str,
    temperature=0.3,
    sample: int | None = None,
    force: bool = False,
):
    exp_A = Path(file_a).stem.replace("results-", "")
    exp_B = Path(file_b).stem.replace("results-", "")

    judge_dir = output_dir / f"{model}_{temperature}".replace(".", "_")
    judge_dir.mkdir(exist_ok=True, parents=True)

    # Check if comparison already exists
    stats_path = judge_dir / f"stats_{exp_A}_vs_{exp_B}.json"
    if not force and stats_path.exists():
        print(f"Comparison between {exp_A} and {exp_B} with {model} (temp={temperature}) already exists. Skipping...")
        return

    # Load the dataframes
    df_a = pd.read_json(file_a, lines=True)
    df_b = pd.read_json(file_b, lines=True)
    comp_df = pd.merge(df_a, df_b, on="text", how="inner", suffixes=["_A", "_B"])[
        ["text", "predicted_triples_A", "predicted_triples_B"]
    ]
    if sample is not None:
        comp_df = comp_df.sample(n=sample)

    # Process the dataframe
    comp_df = process_dataframe(comp_df, model=model, temperature=temperature)

    # Save intermediate results
    output_path = judge_dir / f"comparison_{exp_A}_vs_{exp_B}_{model}_{temperature}.jsonl"
    comp_df.to_json(output_path, orient="records", lines=True)

    # Save stats
    stats = compute_stats(comp_df)
    stats_with_meta = {
        "model_a": exp_A,
        "model_b": exp_B,
        "llm.model": model,
        "llm.temperature": temperature,
        "sample_size": len(comp_df),
        "stats": stats,
    }
    with open(stats_path, "w") as f:
        json.dump(stats_with_meta, f, indent=2)


@app.command()
def compare(
    input_path: str = typer.Argument(..., help="Path to directory containing result files"),
    pattern: str = typer.Option("*.jsonl", help="Glob pattern for result files"),
    out: str = typer.Option("comparisons", help="Directory to save comparison results"),
    sample: int | None = typer.Option(None, "--sample", "-s", help="Number of samples to sample"),
    model: str = typer.Option("qwen-2.5-32b", "--model", "-m", help="Model to use"),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Temperature for sampling"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recomputation of existing comparisons"),
):
    input_path = Path(input_path)
    out = Path(out)
    out.mkdir(exist_ok=True)

    # Find all files matching the pattern
    files = list(input_path.glob(pattern))
    print(f"Found {len(files)} files matching pattern {pattern}")

    # Compare all pairs
    for file_a, file_b in tqdm(list(itertools.combinations(files, 2))):
        print(f"\nComparing {file_a.stem} vs {file_b.stem}")
        compare_pair(file_a, file_b, out, model=model, temperature=temperature, sample=sample, force=force)


@app.command()
def leaderboard(
    comparisons_dir: str = typer.Argument(..., help="Directory containing comparison results"),
    out: str = typer.Option("leaderboard.csv", help="Output file for leaderboard"),
):
    comparisons_dir = Path(comparisons_dir)

    # Load all stats files
    stats_files = list(comparisons_dir.glob("**/stats_*.json"))
    all_results = []
    for stats_file in stats_files:
        with open(stats_file) as f:
            all_results.append(json.load(f))

    # Build leaderboard
    all_models: Set[str] = set()
    for result in all_results:
        all_models.update([result["model_a"], result["model_b"]])

    scores = {model: 0 for model in all_models}
    total_samples = {model: 0 for model in all_models}

    for result in all_results:
        stats = result["stats"]
        model_a = result["model_a"]
        model_b = result["model_b"]
        sample_size = result["sample_size"]

        # Handle A/B/DRAW format
        a_wins = stats.get("A", 0)
        b_wins = stats.get("B", 0)
        draws = stats.get("DRAW", 0)

        # Add scores
        scores[model_a] += a_wins
        scores[model_b] += b_wins
        # Each model gets 0.5 points for a draw
        scores[model_a] += draws * 0.5
        scores[model_b] += draws * 0.5

        # Update total samples
        total_samples[model_a] += sample_size
        total_samples[model_b] += sample_size

    # Calculate weighted average scores
    weighted_scores = {
        model: scores[model] / total_samples[model] if total_samples[model] > 0 else 0 for model in all_models
    }

    # Sort and save leaderboard
    out = Path(out)
    out.parent.mkdir(exist_ok=True, parents=True)

    leaderboard_df = pd.DataFrame(
        sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True),
        columns=["name", "score"],
    )
    leaderboard_df.to_csv(out, index=False)
    leaderboard_df.to_json(out.with_suffix(".json"), orient="records")


if __name__ == "__main__":
    app()
