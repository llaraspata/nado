#!/usr/bin/env bash
# Runs NADO on the toxicity task (RealToxicityPrompts) locally, for every base lm in MODELS (config.yaml entries):
# samples from the base lm, trains R_c, generates continuations of the test prompts and computes the metrics
# (train_toxicity.py). Each model runs in the venv of its config.yaml entry (.venv for gpt2, .venv-new for the newer llms).
#
# Needs the data and models downloaded by recas/setup.sh, and the fudge repo next to this one, linked as ./fudge
# (ln -s ../fudge fudge, also done by setup.sh), for the FUDGE toxicity predictors used as oracle (fudge/ckpt/toxicity/<family>).
# Models whose predictor isn't there (e.g. llama32 until it's trained in fudge) are skipped.
#
# Usage: ./run_toxicity.sh
#
# Everything of a run goes to logs/toxicity/<RUN_ID>/:
#   run.log                          the whole output of this script
#   train_<model>_<tag>.log          output of train_toxicity.py (sampling, training, validation, metrics)
#   preds_<model>_<tag>.log          generations, one per line, aligned with the test prompts
#   stats_<model>_<tag>.json         metrics, model statistics, resources used and arguments of the run
#   summary.txt                      which models succeeded, were skipped or failed
# Samples and R_c checkpoints are kept in dump/toxicity/<model>/ and reused by later runs with the same settings.
#
# Environment variables:
#   MODELS              config.yaml model entries to run (default: all of them)
#   NUM_TEST_PROMPTS    only use the first N test prompts (default: all ~20k)
#   TARGET              nontoxic (default) or toxic: label the oracle requires
#   CONSTRAINT_FACTOR   weight of R_c at decoding (default: 1.0; 0 = plain greedy decoding with the base lm, no NADO)
#   EXTRA_ARGS          more arguments for train_toxicity.py, e.g. "--num_epochs 3 --num_train_prompts 5000"
#   RUN_ID              name of the run's log dir (default: current date and time)
#   CUDA_VISIBLE_DEVICES  gpu to use, numbered as in nvidia-smi (default: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODELS="${MODELS:-gpt2_medium llama32_1b llama32_3b gemma4_e2b gemma4_e4b qwen25_0_5b qwen25_1_5b qwen25_3b}"
NUM_TEST_PROMPTS="${NUM_TEST_PROMPTS:-}"
TARGET="${TARGET:-nontoxic}"
CONSTRAINT_FACTOR="${CONSTRAINT_FACTOR:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID # so gpu numbers are the ones nvidia-smi shows
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="logs/toxicity/$RUN_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

echo "== run $RUN_ID: models [$MODELS], target $TARGET, constraint factor $CONSTRAINT_FACTOR, test prompts ${NUM_TEST_PROMPTS:-all}, gpu $CUDA_VISIBLE_DEVICES"
echo "== logs in $LOG_DIR"

if [ ! -f data/RealToxicityPrompts/prompts.jsonl ]; then
    echo "ERROR: data/RealToxicityPrompts/prompts.jsonl not found: run recas/setup.sh"
    exit 1
fi

entry() { # entry <model> <key>: value of a key of a config.yaml model entry, empty if missing
    .venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['models'][sys.argv[1]].get(sys.argv[2]) or '')" "$1" "$2"
}

declare -A STATUS # model entry -> result, for the summary
for model in $MODELS; do
    venv=$(entry "$model" venv)
    venv=${venv:-.venv}
    oracle=$(entry "$model" oracle_ckpt)
    if [ ! -x "$venv/bin/python" ]; then
        echo "WARNING: $venv not found, skipping $model (see README for how to create it)"
        STATUS[$model]="skipped (no $venv)"
        continue
    fi
    if [ ! -f "$oracle" ]; then
        echo "WARNING: oracle $oracle not found, skipping $model: train the toxicity predictor in fudge first"
        STATUS[$model]="skipped (no oracle)"
        continue
    fi

    tag="${TARGET}_lambda${CONSTRAINT_FACTOR}"
    args=(--model "$model" --target "$TARGET" --constraint_factor "$CONSTRAINT_FACTOR"
          --preds_file "$LOG_DIR/preds_${model}_$tag.log" --stats_file "$LOG_DIR/stats_${model}_$tag.json")
    [ -n "$NUM_TEST_PROMPTS" ] && args+=(--num_test_prompts "$NUM_TEST_PROMPTS")

    echo "== $model ($venv): log in $LOG_DIR/train_${model}_$tag.log"
    # shellcheck disable=SC2086 # EXTRA_ARGS is split into arguments on purpose
    if "$venv/bin/python" -u train_toxicity.py "${args[@]}" $EXTRA_ARGS > "$LOG_DIR/train_${model}_$tag.log" 2>&1; then
        STATUS[$model]="done"
        sed -n '/^num_generations/,/^perplexity/p' "$LOG_DIR/train_${model}_$tag.log"
    else
        echo "ERROR: $model failed, see $LOG_DIR/train_${model}_$tag.log"
        STATUS[$model]="failed"
    fi
done

{
    echo "run $RUN_ID: target $TARGET, constraint factor $CONSTRAINT_FACTOR, test prompts ${NUM_TEST_PROMPTS:-all}"
    for model in $MODELS; do
        printf '%-14s %s\n' "$model" "${STATUS[$model]:-not run}"
    done
} | tee "$LOG_DIR/summary.txt"
