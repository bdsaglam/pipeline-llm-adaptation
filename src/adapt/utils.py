import os
import random

import dspy
import numpy as np


def set_seed(seed):
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def configure_lm(model, temperature):
    lm = dspy.LM(
        "openai/" + model,
        temperature=temperature,
        cache=False,
        api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm)


def dynamic_import(module, name):
    import importlib

    return getattr(importlib.import_module(module), name)


def dedup(items, key=lambda x: x):
    seen = set()
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            yield item
