#!/bin/sh

RESULTS_PATH="data/generated/evaluation/results.jsonl"

get_results() {
    local exp_name="$1"
    dvc experiments apply "$exp_name"
    cp "$RESULTS_PATH" "tmp/results-${exp_name}.jsonl"
    git reset --hard HEAD
    dvc checkout
}

exp_names=($@)
for exp_name in "${exp_names[@]}"; do
    get_results "$exp_name"
done