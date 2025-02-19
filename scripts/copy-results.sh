#!/bin/sh

RESULTS_PATH="data/generated/evaluation/results.jsonl"

get_results() {
    local exp_name="$1"
    dvc experiments apply "$exp_name"
    cp "$RESULTS_PATH" "tmp/results-${exp_name}.jsonl"
    git reset --hard HEAD
    dvc checkout
}

for exp_name in "$@"; do
    get_results "$exp_name"
done