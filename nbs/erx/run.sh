#!/bin/sh

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='bfsrs-ulti' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='bfsrs-medium' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='noop' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='bfsrs-ulti' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='bfsrs-high' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='miprov2-medium' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='miprov2-light' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.5' \
    -S run='1' \
    -S train.optimizer='bfsrs-medium' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='noop' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='bfsrs-high' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='miprov2-medium' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

dvc exp run --queue \
    -S task='erx' \
    -S train.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S train.dataset.name='release_v3.0_en' \
    -S train.dataset.split='"train[:100]"' \
    -S evaluation.dataset.path='bdsaglam/web_nlg-erx-concat' \
    -S evaluation.dataset.name='release_v3.0_en' \
    -S evaluation.dataset.split='"dev"' \
    -S train.ensemble='no' \
    -S lm.temperature='0.0' \
    -S run='1' \
    -S train.optimizer='miprov2-light' \
    -S program.prompting='structured' \
    -S lm.model='llama-3-8b'

