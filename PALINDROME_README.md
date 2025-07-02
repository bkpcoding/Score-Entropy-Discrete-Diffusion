# Byte-Level Palindrome Diffusion Language Model

This implementation extends the SEDD (Score Entropy Discrete Diffusion) framework to generate palindromes using byte-level processing. The byte-level approach provides character-level control, making it ideal for enforcing palindromic constraints.

## Key Features

- **Byte-level processing**: Works with raw bytes (0-255) plus special tokens
- **Palindrome constraints**: Built-in constraints during training and sampling
- **Structured generation**: Specifically designed for palindromic text
- **Flexible length**: Supports palindromes of various lengths up to 128 characters

## Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate sedd
```

### 2. Test the Implementation
```bash
# Test palindrome processing (no model required)
python run_palindrome_sample.py

# Test dataset creation
python -c "from byte_palindrome_data import *; create_byte_palindrome_dataset(size=10)"
```

### 3. Train the Palindrome Model
```bash
# Train with default settings
python train_palindrome.py

# Train with custom settings
python train_palindrome.py training.batch_size=32 model.hidden_size=256
```

### 4. Generate Palindromes
```bash
# Generate palindromes from trained model
python run_palindrome_sample.py --model_path checkpoints/best_model.pth --num_samples 10

# Generate with different constraint strength
python run_palindrome_sample.py --constraint_strength 0.9 --num_samples 20

# Run comprehensive evaluation
python run_palindrome_sample.py --evaluate --num_samples 50
```

## Architecture Overview

### Byte-Level Vocabulary
- **Size**: 259 tokens (256 bytes + PAD + BOS + EOS)
- **Special tokens**:
  - `PAD (256)`: Padding token
  - `BOS (257)`: Beginning of sequence
  - `EOS (258)`: End of sequence

### Model Configuration
- **Hidden size**: 512 (optimized for byte-level)
- **Sequence length**: 128 characters max
- **Attention heads**: 8
- **Layers**: 8 transformer blocks
- **Vocabulary**: 259 tokens

### Palindrome Constraints
1. **Symmetric position averaging**: During training, symmetric positions are encouraged to have similar distributions
2. **Hard projection**: During sampling, discrete tokens are projected to maintain palindromic structure
3. **Constraint strength**: Adjustable parameter (0.0 to 1.0) controlling how strictly palindromes are enforced

## File Structure

```
├── byte_palindrome_data.py      # Byte-level data processing and dataset creation
├── palindrome_sampling.py       # Palindrome-constrained sampling strategies
├── train_palindrome.py         # Training script for palindrome model
├── run_palindrome_sample.py    # Generation and evaluation script
├── configs/palindrome_byte.yaml # Configuration for palindrome model
└── PALINDROME_README.md        # This documentation
```

## Configuration Options

### Training Parameters
```yaml
training:
  batch_size: 64          # Batch size for training
  n_iters: 100000        # Number of training iterations
  lr: 1e-4               # Learning rate (lower for byte-level)
  
model:
  hidden_size: 512       # Model hidden dimension
  length: 128            # Maximum sequence length
  n_blocks: 8            # Number of transformer blocks
  n_heads: 8             # Number of attention heads
```

### Sampling Parameters
```yaml
sampling:
  predictor: analytic    # Sampling strategy
  steps: 64              # Number of diffusion steps
  noise_removal: true    # Apply final denoising
```

## Examples of Generated Palindromes

The model can generate various types of palindromes:

### Single Words
- `level`
- `radar`
- `civic`
- `rotor`

### Phrases  
- `A man a plan a canal Panama`
- `Was it a car or a cat I saw`
- `Never odd or even`
- `Do geese see God`

### Character-level
- `abccba`
- `12321`
- `xyzzyx`

## Evaluation Metrics

- **Palindrome Rate**: Percentage of generated samples that are valid palindromes
- **Average Length**: Mean character length of generated palindromes
- **Diversity**: Variety in generated palindromic structures

## Advanced Usage

### Custom Constraint Strength
```python
from palindrome_sampling import sample_palindromes

# Weak constraints (more diverse, fewer valid palindromes)
results = sample_palindromes(model, graph, noise, constraint_strength=0.3)

# Strong constraints (more palindromes, less diversity)
results = sample_palindromes(model, graph, noise, constraint_strength=0.9)
```

### Custom Dataset
```python
from byte_palindrome_data import create_byte_palindrome_dataset

# Create custom palindrome dataset
palindromes = create_byte_palindrome_dataset(size=5000, max_length=64)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size in config
2. **Low palindrome rate**: Increase constraint_strength or train longer
3. **Empty generations**: Check model loading and ensure proper training

### Performance Tips

1. **Start small**: Begin with shorter sequences (64 chars) and smaller models
2. **Monitor palindrome rate**: Track during training for early stopping
3. **Adjust constraints**: Balance between palindrome validity and diversity

## Research Applications

This implementation is suitable for:
- Studying structured text generation
- Exploring constraint satisfaction in language models
- Investigating byte-level vs token-level modeling
- Developing controllable text generation systems

## Citation

Based on the SEDD framework:
```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```