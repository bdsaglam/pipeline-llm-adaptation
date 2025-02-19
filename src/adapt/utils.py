import os
import random

import dspy
import numpy as np


def set_seed(seed):
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def make_openai_lm(model, temperature: float = 0.0, cache=False, api_base=None, api_key=None, **kwargs):
    return dspy.LM(
        "openai/" + model,
        temperature=temperature,
        cache=cache,
        api_base=api_base or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


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
