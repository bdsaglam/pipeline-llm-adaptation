import os
import re

import dspy
import pandas as pd
import weave
from datasets import load_dataset
from dspy.primitives.prediction import Prediction
from fuzzywuzzy import fuzz


def parse_triples(completion: str):
    return [triple.strip() for triple in completion.split("\n") if triple.strip()]


class EntityRelationExtraction(dspy.Signature):
    """Extract `subject | predicate | object` triples from text."""

    text: str = dspy.InputField()
    triples: list[str] = dspy.OutputField(
        desc='The triples extracted from the text. Each triple should be in the format "subject | predicate | object".'
    )


class SFTPredict:
    def __init__(self):
        self.lm = None

    def set_lm(self, lm):
        self.lm = lm

    def get_lm(self):
        return self.lm

    def load(self, path):
        pass

    def __call__(self, **kwargs):
        text = kwargs.pop("text")
        lm = kwargs.pop("lm", self.lm) or dspy.settings.lm
        messages = [
            {"role": "user", "content": text},
        ]
        responses = lm(messages=messages)
        triples = parse_triples(responses[0])
        return Prediction(triples=triples)


def make_program(prompting: str):
    if prompting == "structured":
        if re.findall("localhost|127.0.0.1|0.0.0.0", os.getenv("OPENAI_BASE_URL", "")):
            from adapt.dspy.tgi_json_adapter import TGIJSONAdapter

            dspy.settings.adapter = TGIJSONAdapter()
        return dspy.Predict(EntityRelationExtraction)
    elif prompting == "sft":
        return SFTPredict()
    else:
        raise ValueError(f"Invalid prompting strategy: {prompting}")


def load_examples(dataset_path: str, dataset_name: str, dataset_split: str):
    ds = load_dataset(dataset_path, dataset_name, split=dataset_split)
    return [dspy.Example(text=x["text"], triples=x["triples"]).with_inputs("text") for x in ds]


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


@weave.op
def evaluate_pred(example, pred, trace=None):
    return compute_scores(pred.triples, example.triples)["fuzzy.f1"]


def make_results_dataframe(results):
    processed_results = []
    for example, pred, _ in results:
        result = {
            "text": example.text,
            "triples": example.triples,
            "predicted_triples": pred.triples,
            **compute_scores(pred.triples, example.triples),
        }
        processed_results.append(result)
    return pd.DataFrame(processed_results)


def aggregate_scores(result_df):
    return (
        result_df[["exact.precision", "exact.recall", "exact.f1", "fuzzy.precision", "fuzzy.recall", "fuzzy.f1"]]
        .mean()
        .to_dict()
    )
