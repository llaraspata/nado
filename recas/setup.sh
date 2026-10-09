#!/bin/bash
# One-off environment setup for the NADO machine translation experiment (train_MT.py) on ReCaS.
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

# Output folders used by train_MT.py
mkdir -p dump/MT dump/reg_MT

echo "Setup completed: $(date)"
