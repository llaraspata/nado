#!/bin/bash
set -e

cd /lustrehome/llaraspata/nado

# the models were downloaded here by recas/setup.sh
export HF_HOME="${HF_HOME:-/lustrehome/llaraspata/nado/.cache/huggingface}"

# venv of the model entry in config.yaml: .venv-new for the newer llm families (llama, qwen, gemma), else .venv
VENV=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['models'][sys.argv[1]].get('venv') or '.venv')" "$1")
source "$VENV/bin/activate"

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
export CUDA_LAUNCH_BLOCKING=1   # surfaces the real CUDA error at the failing call

# $1 model entry in config.yaml, $2 output log; further arguments go to train_toxicity.py (e.g. --num_test_prompts 1000)
python -u train_toxicity.py --model "$1" "${@:3}" > "$2"
