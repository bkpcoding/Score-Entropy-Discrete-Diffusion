#!/bin/bash
#SBATCH --job-name=palindrome_4gpu
#SBATCH --partition=spgpu2
#SBATCH --account=bucherb_owned1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=palindrome_train_%j.out
#SBATCH --error=palindrome_train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda module
module load mamba/py3.10

# Activate the sedd environment
conda init
source ~/.bashrc
conda activate sedd

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Enable wandb logging
export USE_WANDB=true
export WANDB_PROJECT="palindrome-diffusion-4gpu"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Wandb enabled: $USE_WANDB"
echo "Starting time: $(date)"

# Test GPU setup
echo "=== GPU Test ==="
python test_multi_gpu.py ngpus=$SLURM_GPUS_PER_NODE
echo "=================="

# Pre-training stage (optional - comment out if already done)
echo "Starting pre-training on Wikipedia..."
python /nfs/turbo/coe-mandmlab/bpatil/projects/Score-Entropy-Discrete-Diffusion/pretrain_byte.py \
    ngpus=$SLURM_GPUS_PER_NODE \
    training.batch_size=1024 \
    training.n_iters=100000 \
    training.disable_checkpoint_loading=false \
    model.hidden_size=384 \
    model.n_blocks=6 \
    model.n_heads=6

echo "Pre-training completed. Starting palindrome fine-tuning..."

# Find the latest pre-training checkpoint
PRETRAIN_CHECKPOINT=$(find checkpoints -name "checkpoint_*.pth" -type f | sort -V | tail -1)
echo "Using pre-training checkpoint: $PRETRAIN_CHECKPOINT"

# Palindrome fine-tuning stage
python /nfs/turbo/coe-mandmlab/bpatil/projects/Score-Entropy-Discrete-Diffusion/train_palindrome.py \
    ngpus=$SLURM_GPUS_PER_NODE \
    training.batch_size=512 \
    training.n_iters=15000 \
    training.snapshot_freq=2000 \
    training.eval_freq=500 \
    training.log_freq=100 \
    training.disable_checkpoint_loading=false \
    training.pretrain_checkpoint="$PRETRAIN_CHECKPOINT" \
    model.hidden_size=384 \
    model.n_blocks=6 \
    model.n_heads=6 \
    optim.lr=1e-4

echo "Job finished at: $(date)"