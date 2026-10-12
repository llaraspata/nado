#!/bin/bash
set -e

cd /lustrehome/llaraspata/nado

source .venv/bin/activate

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
export CUDA_LAUNCH_BLOCKING=1   # surfaces the real CUDA error at the failing call


python -u python train_MT.py > "$1"