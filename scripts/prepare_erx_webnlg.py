import random
import re
from typing import Generator, TypeVar

import numpy as np
import typer
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import repo_exists

load_dotenv()


app = typer.Typer()


def split_camel_case(input_str):
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", input_str)
    return [m.group(0) for m in matches]


def set_seed(seed):
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


SEED = 89
set_seed(SEED)


def _transform_relation(relation: str):
    return " ".join([word.lower() for word in split_camel_case(relation)]).strip()


def _transform_entity(entity: str):
    return entity.replace("_", " ").strip()


def _transform_triple(triple_string: str):
    delimiter = " | "
    triple_string = triple_string.replace('"', "").replace("''", "")
    subject, relation, obj = triple_string.split(delimiter)
    relation = _transform_relation(relation)
    subject = _transform_entity(subject)
    obj = _transform_entity(obj)
    return delimiter.join([subject, relation, obj])


def _batch_transform(examples):
    for eid, lex, mts in zip(examples["eid"], examples["lex"], examples["modified_triple_sets"]):
        for text in lex["text"]:
            triples = [_transform_triple(triplet_string) for triplet_string in mts["mtriple_set"][0]]
            yield dict(eid=eid, text=text, triples=triples)


def batch_transform(examples):
    records = list(_batch_transform(examples))
    return {
        "eid": [record["eid"] for record in records],
        "text": [record["text"] for record in records],
        "triples": [record["triples"] for record in records],
    }


T = TypeVar("T")


def chunk_random(lst: list[T], min_chunk: int = 2, max_chunk: int = 4) -> Generator[list[T], None, None]:
    if len(lst) < min_chunk:
        yield lst
        return

    i = 0
    while i < len(lst):
        if len(lst) - i < min_chunk:
            break
        chunk_size = random.randint(min_chunk, min(max_chunk, len(lst) - i))
        yield lst[i : i + chunk_size]
        i += chunk_size


def chunk_random_dataset(ds, min_chunk=1, max_chunk=3):
    for indices in chunk_random(range(len(ds)), min_chunk, max_chunk):
        yield ds.select(indices).to_list()


def concat_examples(examples):
    return {
        "eid": "\n".join([example["eid"] for example in examples]),
        "text": "\n".join([example["text"] for example in examples]),
        "triples": [triple for example in examples for triple in example["triples"]],
    }


def create_erx_datasets(dsd, erx_dataset_path, dataset_name):
    erx_dsd = dsd.map(batch_transform, batched=True, remove_columns=dsd["train"].column_names)

    assert "eid" in erx_dsd["train"].features
    assert "text" in erx_dsd["train"].features
    assert "triples" in erx_dsd["train"].features
    assert isinstance(erx_dsd["train"][0]["triples"], list)
    assert isinstance(erx_dsd["train"][0]["triples"][0], str)
    print(f"Pushing {erx_dataset_path} to the hub")
    erx_dsd.push_to_hub(erx_dataset_path, config_name=dataset_name)
    return erx_dsd


def create_concat_erx_datasets(erx_dsd, concat_erx_dataset_path, dataset_name):
    concat_erx_dsd = DatasetDict(
        {
            k: Dataset.from_list(
                [{"chunk": chunk} for chunk in chunk_random_dataset(erx_dsd.shuffle(SEED), min_chunk=1, max_chunk=7)]
            )
            for k, ds in erx_dsd.items()
        }
    ).map(lambda x: concat_examples(x["chunk"]), remove_columns=["chunk"])
    print(f"Pushing {concat_erx_dataset_path} to the hub")
    concat_erx_dsd.push_to_hub(concat_erx_dataset_path, config_name=dataset_name)
    return concat_erx_dsd


def make_openai_chat(example):
    triples_str = "\n".join(example["triples"])
    return {
        "messages": [
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": triples_str},
        ],
    }


def describe_dataset_dict(dsd: DatasetDict):
    print("Features:", list(dsd["train"].features.keys()))
    split_descriptions = []
    for split in dsd:
        ds = dsd[split]
        split_descriptions.append(f"{split}={len(ds)}")
    print("Splits:", " ".join(split_descriptions))
    print("-" * 120)


@app.command()
def main(
    hf_username: str = "bdsaglam",
    overwrite: bool = typer.Option(False, help="Overwrite existing dataset on the hub"),
):
    print("-" * 120)
    dataset_path = "web_nlg"
    dataset_name = "release_v3.0_en"
    dsd = load_dataset(dataset_path, dataset_name, trust_remote_code=True)
    print(f"Dataset: {dataset_path}/{dataset_name}")
    describe_dataset_dict(dsd)

    erx_dataset_path = f"{hf_username}/{dataset_path.split('/', 1)[-1]}-erx"
    if not repo_exists(erx_dataset_path, repo_type="dataset") or overwrite:
        erx_dsd = create_erx_datasets(dsd, erx_dataset_path, dataset_name)
    else:
        erx_dsd = load_dataset(erx_dataset_path, dataset_name, trust_remote_code=True)

    print(f"Dataset: {erx_dataset_path}")
    describe_dataset_dict(erx_dsd)

    concat_erx_dataset_path = erx_dataset_path + "-concat"
    if not repo_exists(concat_erx_dataset_path, repo_type="dataset") or overwrite:
        concat_erx_dsd = create_concat_erx_datasets(erx_dsd, concat_erx_dataset_path, dataset_name)
    else:
        concat_erx_dsd = load_dataset(concat_erx_dataset_path, dataset_name, trust_remote_code=True)

    print(f"Dataset: {concat_erx_dataset_path}")
    describe_dataset_dict(concat_erx_dsd)

    chat_erx_dataset_path = concat_erx_dataset_path + "-chat"
    if not repo_exists(chat_erx_dataset_path, repo_type="dataset") or overwrite:
        chat_erx_dsd = concat_erx_dsd.map(
            make_openai_chat,
            batched=True,
            remove_columns=concat_erx_dsd["train"].column_names,
        )
        chat_erx_dsd.push_to_hub(chat_erx_dataset_path, config_name=dataset_name)
    else:
        chat_erx_dsd = load_dataset(chat_erx_dataset_path, dataset_name, trust_remote_code=True)

    print(f"Dataset: {chat_erx_dataset_path}")
    describe_dataset_dict(chat_erx_dsd)


if __name__ == "__main__":
    app()
