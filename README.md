# Palindrome Generation with Score-Entropy Discrete Diffusion

![Palindromes_Diffusion](./palindromes_diffusion.gif)

This directory contains code for training and sampling palindrome generation models using the Score-Entropy Discrete Diffusion (SEDD) framework. The workflow involves data preparation, pre-training on Wikipedia text, fine-tuning on palindrome data, and sampling palindromes from the trained model.


## Complete Workflow

### Environment Creation
Use uv to install dependencies
```bash
# create conda environment
conda create --name sedd python=3.10
# install pytorch and dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# install the requirements
uv pip install -r requirements.txt --no-build-isolation
```

### 1. Data Preparation

First, prepare the byte-level Wikipedia dataset for pre-training, this will create two binary files for faster loading during training.

```bash
python -m palindromes.prepare_byte_data --dataset_percent 10.0 --block_size 32
```

**Options:**
- `--dataset_percent`: Percentage of Wikipedia dataset to use (default: 20.0)
- `--block_size`: Sequence length for training (default: 32)
- `--num_proc`: Number of processes for parallel processing (default: 8)
- `--cache_dir`: Directory for caching data (default: data)

**Output files:**
- `data/train_filter.bin`: Training data
- `data/val_filter.bin`: Validation data

### 2. Pre-training on Wikipedia

Pre-train the model on Wikipedia text to learn general language patterns, the exact config can be found in `../configs/pretrain_byte.yaml`

```bash
python -m palindromes.pretrain
```

**Key parameters in `configs/pretrain_byte.yaml`:**
- `training.batch_size`: Batch size (default: 256)
- `training.n_iters`: Number of training iterations (default: 20000)
- `model.hidden_size`: Hidden dimension size (default: 768)
- `model.length`: Sequence length (default: 32)

### 3. Fine-tuning on Palindromes

Fine-tune the model on palindrome data:

```bash
python -m palindromes.finetune
```

**To use a pre-trained checkpoint:**
```bash
python -m palindromes.finetune training.pretrain_checkpoint=checkpoints/checkpoints/checkpoint_4.pth
```

### 4. Generate Palindromes

Generate palindromes from the trained model:

```bash
python -m palindromes.run_palindrome_sample \
    --model_path palindromes/checkpoints/checkpoints/finetune_checkpoint_71.pth \
    --num_samples 50 \
    --max_length 16 \
    --steps 100
```

**Options:**
- `--model_path`: Path to trained model checkpoint
- `--num_samples`: Number of palindromes to generate (default: 100)
- `--max_length`: Maximum length of palindromes (default: 12)
- `--steps`: Number of diffusion steps (default: 100)
- `--device`: Device to use (default: cuda)

## SLURM Job Submission

For high-performance computing clusters, use the provided SLURM script:

```bash
sbatch submit_palindrome_train.slurm
```

The script performs the complete workflow:
1. Set up the conda environment
2. Run pre-training on Wikipedia
3. Find the latest pre-training checkpoint
4. Run fine-tuning with the pre-trained checkpoint
5. Use 4 GPUs by default

**Key SLURM parameters:**
- `--gpus-per-node=4`: Number of GPUs
- `--time=48:00:00`: Maximum job time (48 hours)
- `--mem=32G`: Memory allocation

## Directory Structure

```
palindromes/
├── README.md                    # This comprehensive guide
├── prepare_byte_data.py         # Data preparation script
├── pretrain.py                  # Pre-training on Wikipedia
├── finetune.py                  # Fine-tuning on palindromes
├── run_palindrome_sample.py     # Sampling and evaluation script
├── sample_training_data.py      # Data analysis and visualization
├── submit_palindrome_train.slurm # SLURM job script
├── palindrome_sampling.py       # Palindrome-constrained sampling
├── palindrome_json_loader.py    # Palindrome data loading
├── byte_data.py                 # Byte-level data processing
├── binary_data_loader.py        # Binary data utilities
├── data/                        # Processed data files
│   ├── train_filter.bin
│   ├── val_filter.bin
│   └── ...
├── checkpoints/                 # Model checkpoints
│   ├── checkpoints/             # Fine-tuning checkpoints
│   ├── checkpoints-meta/        # Meta checkpoints
│   └── ...
└── configs/                     # Configuration files (in parent directory)
    ├── pretrain_byte.yaml
    └── palindrome_byte.yaml
```

## Configuration Files

### Pre-training Config (`configs/pretrain_byte.yaml`)

```yaml
training:
  batch_size: 256         # Large batch for pre-training
  n_iters: 20000         # Pre-training iterations
  snapshot_freq: 5000    # Checkpoint frequency
  lr: 3e-4               # Learning rate

model:
  hidden_size: 768       # Model dimension
  length: 32             # Sequence length
  n_blocks: 12           # Transformer layers
  n_heads: 12            # Attention heads

data:
  train: byte_wikipedia  # Wikipedia dataset
  cache_dir: data        # Data cache directory
```

### Fine-tuning Config (`configs/palindrome_byte.yaml`)

```yaml
training:
  batch_size: 64         # Smaller batch for fine-tuning
  n_iters: 75000         # Fine-tuning iterations
  snapshot_freq: 1000    # More frequent checkpoints
  pretrain_checkpoint: null # Pre-trained model path

model:
  hidden_size: 768       # Match pre-training dimensions
  length: 32             # Sequence length

data:
  train: byte_palindrome # Palindrome dataset
  cache_dir: data
```

## Model Architecture

The model uses a transformer-based architecture with:
- **Vocabulary Size**: 259 tokens (256 byte values + PAD + BOS + EOS)
- **Hidden Size**: 768 (configurable)
- **Attention Heads**: 12 (configurable)
- **Layers**: 12 transformer blocks (configurable)
- **Sequence Length**: 32 characters (configurable)

### Special Tokens
- `PAD (256)`: Padding token
- `BOS (257)`: Beginning of sequence
- `EOS (258)`: End of sequence

## Data Analysis

Analyze the training data to understand what the model learns:

```bash
python sample_training_data.py --num_batches 5 --max_samples 1000
```

This creates visualizations showing:
- Byte value distributions
- Sequence length statistics
- Character frequency analysis
- Training batch compositions

## Training Process

### 1. Data Processing
- Wikipedia text is cleaned and filtered
- Text is converted to byte sequences
- Sequences are chunked to fixed length with padding
- BOS/EOS tokens are added

### 2. Pre-training
- Model learns general language patterns from Wikipedia
- Uses larger batch sizes (256) and longer training (20k iterations)
- Saves checkpoints every 5000 steps
- Generates sample text during training

### 3. Fine-tuning
- Loads pre-trained checkpoint (optional but recommended)
- Trains on palindrome-specific data
- Uses smaller batch sizes (64) for specialized task
- Saves checkpoints every 1000 steps
- Generates sample palindromes during training

### 4. Sampling
- Uses trained model to generate palindromes
- Applies diffusion process for controlled generation
- Evaluates palindrome validity and quality

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Ensure the model path is correct and relative to your working directory
2. **CUDA out of memory**: Reduce batch size in config files
3. **Data loading errors**: Check that data preparation completed successfully
4. **Import errors**: Ensure all dependencies are installed and Python paths are correct
5. **Low palindrome rate**: Train longer or adjust sampling parameters

### Memory Management

For large models or limited GPU memory:
- Reduce `training.batch_size` in config files
- Use gradient accumulation (`training.accum > 1`)
- Enable mixed precision training
- Use smaller model dimensions (`model.hidden_size`)
