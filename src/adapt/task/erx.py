import dspy
import pandas as pd
from datasets import load_dataset
from fuzzywuzzy import fuzz


class EntityRelationExtraction(dspy.Signature):
    """Extract `subject | predicate | object` triples from text."""

    text: str = dspy.InputField()
    triples_str: str = dspy.OutputField(
        desc='The triples extracted from the text. Each triple should be in the format "subject | predicate | object". Triples should be separated by newlines.'
    )


def make_program():
    return dspy.Predict(EntityRelationExtraction)


def load_examples(dataset_path: str, dataset_name: str, dataset_split: str):
    ds = load_dataset(dataset_path, dataset_name, split=dataset_split)
    return [dspy.Example(text=x["text"], triples=x["triples"]).with_inputs("text") for x in ds]


def parse_triples(triples_str: str):
    return [triple.strip() for triple in triples_str.split("\n") if triple.strip()]


def compute_generalized_scores(pred_triples, reference_triples, match_function):
    """Compute precision, recall, and F1-score using a customizable match function."""
    pred_set = set(pred_triples)
    reference_set = set(reference_triples)

    if not pred_set and not reference_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_set or not reference_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = sum(any(match_function(pred, ref) for ref in reference_set) for pred in pred_set)

    precision = true_positives / len(pred_set)
    recall = true_positives / len(reference_set)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def fuzzy_match(pred: str, ref: str) -> bool:
    return fuzz.ratio(pred, ref) > 80


def compute_scores(pred_triples, reference_triples):
    exact_scores = {
        f"exact.{k}": v
        for k, v in compute_generalized_scores(pred_triples, reference_triples, lambda x, y: x == y).items()
    }
    fuzzy_scores = {
        f"fuzzy.{k}": v for k, v in compute_generalized_scores(pred_triples, reference_triples, fuzzy_match).items()
    }
    return {**exact_scores, **fuzzy_scores}


def evaluate_pred(example, pred, trace=None):
    return compute_scores(parse_triples(pred.triples_str), example.triples)["fuzzy.f1"]


def make_results_dataframe(results):
    processed_results = []
    for example, pred, _ in results:
        result = {
            "text": example.text,
            "triples": example.triples,
            "predicted_triples": parse_triples(pred.triples_str),
            **compute_scores(parse_triples(pred.triples_str), example.triples),
        }
        processed_results.append(result)
    return pd.DataFrame(processed_results)


def aggregate_scores(result_df):
    return (
        result_df[["exact.precision", "exact.recall", "exact.f1", "fuzzy.precision", "fuzzy.recall", "fuzzy.f1"]]
        .mean()
        .to_dict()
    )
