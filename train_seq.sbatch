#!/bin/bash
# SoC Compute Cluster job: sequence_model.py on GPU, one seed per array task.
#
# One-time setup, from an SoC login node (after `ssh you@xlogin.comp.nus.edu.sg`):
#   1. Copy this repo over, e.g. from your local machine:
#        rsync -avz --exclude .venv /Users/gaozilin/Desktop/track2/ \
#            you@xlogin.comp.nus.edu.sg:~/track2/
#      (KuaiRand-Pure/data/ is ~194MB — included in that sync; re-download instead
#      if you'd rather not copy it, see README.md.)
#   2. On the cluster, build a GPU-enabled venv (CUDA wheel, NOT the CPU wheel
#      used in the local .venv/ — different index URL):
#        cd ~/track2
#        python3 -m venv .venv_gpu
#        .venv_gpu/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cu121
#   3. Submit: sbatch train_seq.sbatch
#      (or `srun --gpus=1 --pty bash` first if you want an interactive session to debug)
#
# --arch bst (Behavior Sequence Transformer) specifically: on this machine's CPU,
# BST ran ~2450s/epoch — ~35x slower than the DIN variant (~70s/epoch), almost
# certainly because PyTorch's CPU attention path isn't efficient for training-mode
# (gradient-requiring) self-attention with small per-head dims (k=16, 4 heads ->
# head_dim=4). GPU should help by more than raw core count would suggest, since
# this is exactly the shape of workload GPUs are architecturally good at. Unpinned
# --gpus=1 (any available GPU, incl. older T4/V100) was fine for DIN; for BST,
# pinning something stronger if queue wait allows is worth it — e.g. --gpus=a100-40
# or --gpus=h100-96 (replace --gpus=1 below), since it's now genuinely
# compute-bound, not just tiny-model overhead. --time is set generously (2h) since
# GPU speedup here is a real unknown until measured — cluster allows up to 3h.

#SBATCH --job-name=seqfm
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/seqfm_%A_%a.out
#SBATCH --error=logs/seqfm_%A_%a.err

set -e
cd "$(dirname "$0")"
mkdir -p logs

# ARCH env var lets you pick din vs bst without editing this file:
#   ARCH=bst sbatch train_seq.sbatch
ARCH="${ARCH:-bst}"
.venv_gpu/bin/python sequence_model.py --arch "$ARCH" --device cuda --seed "${SLURM_ARRAY_TASK_ID}"
