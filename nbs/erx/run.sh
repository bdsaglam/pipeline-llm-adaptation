#!/bin/sh

dvc exp run --queue \
    -S run=1 \
    -S train.optimizer='miprov2-light' \
    -S train.ensemble='no' \
    -S lm.model='llama-3-8b' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='train[:100]' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='dev'

dvc exp run --queue \
    -S run=1 \
    -S train.optimizer='noop' \
    -S train.ensemble='no' \
    -S lm.model='llama-3-8b' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='train[:100]' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='dev'

dvc exp run --queue \
    -S run=1 \
    -S train.optimizer='bfsrs-medium' \
    -S train.ensemble='no' \
    -S lm.model='llama-3-8b' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='train[:100]' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='dev'

