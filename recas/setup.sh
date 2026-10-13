#!/bin/bash
# One-off environment setup on ReCaS for the NADO experiments: machine translation formality (train_MT.py)
# and toxicity with GPT-2 and the newer llms (train_toxicity.py).
# Run through HTCondor with recas/setup.sub, or directly from the repository root.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"
if [ ! -f train_MT.py ]; then
    echo "train_MT.py not found in $REPO_DIR: run from the repository root or set REPO_DIR" >&2
    exit 1
fi

echo "Host: $(hostname)  Dir: $REPO_DIR  Date: $(date)"

# Keep the Hugging Face cache on lustre so the training jobs reuse the downloaded models
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$REPO_DIR/.cache/nltk_data}"
mkdir -p "$HF_HOME" "$NLTK_DATA" logs

# ---------------------------------------------------------------------------
# 1. Python environment (uv + Python 3.10, as in the README)
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d .venv ]; then
    uv venv --python 3.10 .venv
fi
source .venv/bin/activate
uv pip install -r requirements.txt

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda build', torch.version.cuda)"
deactivate

# .venv-new: transformers 5, needed by the newer llms of the toxicity task (train_toxicity.py: llama 3.2, gemma 4, qwen2.5).
# torch from the pytorch CUDA 12.6 index (NEW_TORCH_INDEX to change it)
NEW_TORCH_INDEX="${NEW_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
if [ ! -d .venv-new ]; then
    uv venv --python 3.11 .venv-new
fi
uv pip install --python .venv-new/bin/python -r requirements-new.txt \
    --index-url "$NEW_TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.venv-new/bin/python -c "import torch, transformers; print('.venv-new: torch', torch.__version__, '| transformers', transformers.__version__, '| cuda build', torch.version.cuda)"
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 2. Data: Fisher-Callhome Spanish-English corpus and fluent references
# ---------------------------------------------------------------------------
# train_MT.py reads fisher-callhome-corpus/corpus/ldc/fisher_{train,dev,test}.es and fisher_train.en
if [ ! -d fisher-callhome-corpus ]; then
    git clone --depth 1 https://github.com/joshua-decoder/fisher-callhome-corpus.git
fi
# train_MT.py reads fluent-fisher/noids/{dev,test}.noid.cleaned_{0,1}
if [ ! -d fluent-fisher ]; then
    git clone --depth 1 https://github.com/isl-mt/fluent-fisher.git
fi

for f in fisher-callhome-corpus/corpus/ldc/fisher_{train,dev,test}.es \
         fisher-callhome-corpus/corpus/ldc/fisher_train.en \
         fluent-fisher/noids/{dev,test}.noid.cleaned_{0,1}; do
    if [ ! -f "$f" ]; then
        echo "WARNING: missing $f (the Spanish side may need LDC access, see README)" >&2
    fi
done

# ---------------------------------------------------------------------------
# 3. FUDGE formality oracle used by NeuralConstraintFunction.init_FUDGE_formality()
# ---------------------------------------------------------------------------
# Expected at ./test_evaluator_gyafc_family_relationships/model.pth.tar. It comes from the
# FUDGE release archive: only ckpt/formality/ is extracted, the rest (poetry, topic, data) is skipped.
FUDGE_EVALUATOR=test_evaluator_gyafc_family_relationships
if [ ! -f "$FUDGE_EVALUATOR/model.pth.tar" ]; then
    mkdir -p .cache/fudge
    if [ ! -f .cache/fudge/large_files.zip ]; then
        wget -O .cache/fudge/large_files.zip https://naacl2021-fudge-files.s3.amazonaws.com/large_files.zip
    fi
    UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o .cache/fudge/large_files.zip 'ckpt/formality/*' -d .cache/fudge

    rm -rf "$FUDGE_EVALUATOR"
    mv ".cache/fudge/ckpt/formality/$FUDGE_EVALUATOR" .
    # Keep the other formality checkpoints (marian_finetune_fisher, predictor_gyafc_entertainment_music)
    mkdir -p ckpt
    rm -rf ckpt/formality
    mv .cache/fudge/ckpt/formality ckpt/formality
    rm -rf .cache/fudge
fi
ls -l "$FUDGE_EVALUATOR"

# ---------------------------------------------------------------------------
# 3b. Toxicity task (train_toxicity.py): RealToxicityPrompts and the FUDGE toxicity oracles
# ---------------------------------------------------------------------------
# train_toxicity.py splits prompts.jsonl into train/val/test itself, as fudge's prepare_toxicity_data.py does
RTP=data/RealToxicityPrompts/prompts.jsonl
if [ ! -f "$RTP" ]; then
    mkdir -p "$(dirname "$RTP")"
    curl -fL --retry 3 -o "$RTP.tmp" https://huggingface.co/datasets/allenai/real-toxicity-prompts/resolve/main/prompts.jsonl
    mv "$RTP.tmp" "$RTP"
fi
echo "$(wc -l < "$RTP") RealToxicityPrompts records in $RTP"

# the oracles are the toxicity predictors trained in the fudge repo (one per tokenizer family), read through ./fudge
if [ ! -e fudge ] && [ -d ../fudge ]; then
    ln -s ../fudge fudge
fi
for ckpt in $(python -c "import yaml; print(' '.join(sorted(set(m['oracle_ckpt'] for m in yaml.safe_load(open('config.yaml'))['models'].values() if m.get('oracle_ckpt')))))"); do
    if [ ! -f "$ckpt" ]; then
        echo "WARNING: missing toxicity oracle $ckpt: train it in the fudge repo (the models using it can't run until then)" >&2
    fi
done

# ---------------------------------------------------------------------------
# 4. Pre-download models so the training jobs do not need network access
# ---------------------------------------------------------------------------
python - <<'EOF'
from transformers import MarianMTModel, MarianTokenizer
import nltk

name = "Helsinki-NLP/opus-mt-es-en"
MarianTokenizer.from_pretrained(name)
MarianMTModel.from_pretrained(name)
nltk.download("punkt", quiet=True)
print("Models cached")
EOF

# Base lms of the toxicity task (every model entry of config.yaml) and gpt2-large, which scores perplexity.
# Llama and Gemma are gated: accept their license on huggingface and set HF_TOKEN (or run `hf auth login`) first,
# otherwise they are skipped with a warning.
.venv-new/bin/python - <<'EOF'
import yaml
from huggingface_hub import snapshot_download

models = yaml.safe_load(open('config.yaml'))['models']
names = sorted(set(m['model_string'] for m in models.values())) + ['openai-community/gpt2-large']
failed = []
for name in names:
    print('downloading', name, flush=True)
    try:
        # original/: llama's duplicate weights in meta's format; the others: duplicate weights (.bin/.pth next to the safetensors) and weights in other frameworks
        snapshot_download(name, ignore_patterns=['original/*', '*.bin', '*.pth', '*.h5', '*.msgpack', '*.ot', '*.onnx', 'onnx/*', '*.tflite', '*.gguf', 'coreml/*'])
    except Exception as e:
        print('WARNING: could not download {} ({}: {}). If it is gated (llama, gemma), accept its license on '
              'huggingface and set HF_TOKEN'.format(name, type(e).__name__, e), flush=True)
        failed.append(name)
print('Toxicity models cached' + (', except ' + ', '.join(failed) if failed else ''))
EOF

# Output folders used by train_MT.py and train_toxicity.py
mkdir -p dump/MT dump/reg_MT dump/toxicity

echo "Setup completed: $(date)"
