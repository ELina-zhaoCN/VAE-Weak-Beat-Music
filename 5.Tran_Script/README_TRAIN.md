# Complete Training Script for Weak Beat Music VAE

Comprehensive training pipeline for training a Variational Autoencoder to generate music with weak/irregular beats.

## Features

✓ **Complete VAE Training**: Reconstruction + KL divergence + Beat loss  
✓ **Warm-up Strategy**: Linear warm-up for β (KL weight) and γ (beat weight)  
✓ **BPM Evaluation**: Autocorrelation-based BPM consistency metrics  
✓ **Checkpointing**: Save best model and periodic checkpoints  
✓ **Tensorboard Logging**: Real-time training visualization  
✓ **Flexible Configuration**: Command-line arguments for all hyperparameters  
✓ **Resume Training**: Continue from saved checkpoints  

## Quick Start

### Basic Training

```bash
python train.py --data_dir ./weak_beat_music --epochs 100 --batch_size 16
```

### With All Options

```bash
python train.py \
  --data_dir ./weak_beat_music \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --latent_dim 128 \
  --kl_weight 1.0 \
  --beat_weight 0.1 \
  --warmup_epochs 20 \
  --eval_bpm \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

## Command-Line Arguments

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./weak_beat_music` | Path to music dataset |
| `--val_split` | `0.2` | Validation set ratio (0-1) |

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--latent_dim` | `128` | Latent space dimension |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-4` | Learning rate (Adam) |
| `--num_workers` | `2` | Data loader workers |

### Loss Weights

| Argument | Default | Description |
|----------|---------|-------------|
| `--kl_weight` | `1.0` | KL divergence weight (β) |
| `--beat_weight` | `0.1` | Beat loss weight (γ) |
| `--warmup_epochs` | `20` | Warmup duration |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | `./checkpoints` | Checkpoint directory |
| `--log_dir` | `./logs` | Tensorboard logs |
| `--save_interval` | `5` | Save every N epochs |
| `--resume` | `None` | Resume from checkpoint |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_bpm` | `False` | Enable BPM evaluation |
| `--no_beat_loss` | `False` | Disable beat loss |

### Other

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | `42` | Random seed |
| `--device` | `auto` | Device (auto/cpu/cuda) |

## Loss Functions

### 1. Reconstruction Loss (MSE)

```python
recon_loss = MSE(reconstruction, target)
```

Measures pixel-wise difference between input and reconstructed Mel-spectrograms.

### 2. KL Divergence Loss

```python
KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

Regularizes latent space to follow standard normal distribution N(0, I).

### 3. Beat Loss

```python
beat_loss = max(autocorr[5:])  # Regularity score
```

Penalizes regular beats by measuring autocorrelation of low-frequency energy envelope.
- **High loss** (0.6-1.0): Regular beats (not desired)
- **Low loss** (0.0-0.3): Irregular/no beats (desired)

### Total Loss

```python
total_loss = recon_loss + β × KL_loss + γ × beat_loss
```

Where β and γ increase linearly during warmup.

## Warm-up Strategy

### Why Warm-up?

Prevents KL collapse and beat loss from dominating early training.

### Schedule

```
Epochs 0-20:  β = β_target × (epoch / 20)
              γ = γ_target × (epoch / 20)
              
Epochs 20+:   β = β_target (fixed)
              γ = γ_target (fixed)
```

**Example** (β_target=1.0, γ_target=0.1):
- Epoch 0: β=0.00, γ=0.00
- Epoch 10: β=0.50, γ=0.05
- Epoch 20+: β=1.00, γ=0.10

## BPM Evaluation

### How It Works

Since we don't have ground truth BPM labels, we use **autocorrelation consistency**:

1. Estimate BPM from original spectrogram
2. Estimate BPM from reconstruction
3. Compare the two estimates

### Metrics

- **BPM MAE**: Mean absolute error between original and reconstructed BPM
- **BPM Consistency**: `1 - (MAE / 100)`, normalized to [0, 1]

### When to Use

```bash
python train.py --eval_bpm  # Enable BPM evaluation
```

**Note**: BPM evaluation is slow (~10-20x slower). Only enable for final evaluation.

## Output Files

### Checkpoints Directory

```
checkpoints/
├── best_model.pt              # Best model (lowest val beat loss)
├── final_model.pt             # Final model after training
├── checkpoint_epoch_5.pt      # Periodic checkpoints
├── checkpoint_epoch_10.pt
├── ...
├── config.json                # Training configuration
└── training_history.json      # Loss history
```

### Checkpoint Contents

```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': {validation_metrics},
    'args': {command_line_args}
}
```

### Tensorboard Logs

```
logs/
└── YYYYMMDD-HHMMSS/
    └── events.out.tfevents...
```

## Monitoring Training

### Console Output

```
Epoch 1/100: 100%|████| 100/100 [00:45<00:00,  2.21it/s, 
    loss=12.3456, recon=10.2345, kl=2.0000, beat=0.1111, β=0.050, γ=0.005]

Epoch 1/100 Summary:
  Train - Total: 12.3456, Recon: 10.2345, KL: 2.0000, Beat: 0.1111
  Val   - Total: 11.9876, Recon: 9.8765, KL: 1.9500, Beat: 0.1611
  ✓ New best model! Val beat loss: 0.1611
```

### Tensorboard

```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006

**Available Plots:**
- Train/val total loss
- Reconstruction loss
- KL divergence loss
- Beat loss
- β and γ schedules
- BPM metrics (if enabled)

## Training Examples

### Example 1: Default Training

```bash
# Train with default settings
python train.py --data_dir ./weak_beat_music
```

### Example 2: Fast Prototyping

```bash
# Quick training with fewer epochs and larger batch
python train.py \
  --data_dir ./weak_beat_music \
  --epochs 20 \
  --batch_size 32 \
  --save_interval 5
```

### Example 3: High Quality Training

```bash
# Longer training with smaller learning rate
python train.py \
  --data_dir ./weak_beat_music \
  --epochs 200 \
  --batch_size 16 \
  --lr 5e-5 \
  --warmup_epochs 40 \
  --eval_bpm
```

### Example 4: Emphasize Beat Loss

```bash
# Higher weight for beat loss
python train.py \
  --data_dir ./weak_beat_music \
  --beat_weight 0.5 \
  --kl_weight 0.5 \
  --warmup_epochs 30
```

### Example 5: Resume Training

```bash
# Continue from checkpoint
python train.py \
  --data_dir ./weak_beat_music \
  --resume ./checkpoints/checkpoint_epoch_50.pt \
  --epochs 100
```

### Example 6: No Beat Loss (Standard VAE)

```bash
# Train standard VAE without beat loss
python train.py \
  --data_dir ./weak_beat_music \
  --no_beat_loss
```

## Complete Training Pipeline

### Step 1: Prepare Data

```bash
# Filter weak beat music from FMA
cd 1.filter_fma_weak_beat
python fma_filter.py --filter --audio-dir ../fma_data/fma_medium
cd ..
```

### Step 2: Train Model

```bash
# Train VAE
python train.py \
  --data_dir ./weak_beat_music \
  --epochs 100 \
  --batch_size 16 \
  --eval_bpm
```

### Step 3: Monitor Training

```bash
# In another terminal
tensorboard --logdir=./logs
```

### Step 4: Evaluate Best Model

```python
import torch
from vae_model import MelSpectrogramVAE

# Load best model
model = MelSpectrogramVAE(latent_dim=128)
checkpoint = torch.load('./checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
samples = model.sample(10, device)
print(f"Generated {samples.shape[0]} samples")
```

## Troubleshooting

### Issue: CUDA out of memory

**Solutions:**
- Reduce batch size: `--batch_size 8`
- Reduce latent dimension: `--latent_dim 64`
- Reduce workers: `--num_workers 0`

### Issue: Loss not decreasing

**Solutions:**
- Check learning rate: try `--lr 5e-5` or `--lr 2e-4`
- Increase warmup: `--warmup_epochs 40`
- Check data loading (should print dataset size)

### Issue: KL collapse (KL loss → 0)

**Solutions:**
- Increase KL weight: `--kl_weight 2.0`
- Longer warmup: `--warmup_epochs 30`
- This is normal in early epochs during warmup

### Issue: Beat loss not decreasing

**Expected**: Beat loss should be low (0.1-0.3) for weak beat music
- If already low: Model is working correctly
- If high (>0.5): Increase `--beat_weight`

### Issue: Slow training

**Solutions:**
- Increase batch size: `--batch_size 32`
- More workers: `--num_workers 4`
- Disable BPM eval: remove `--eval_bpm`
- Use GPU: automatic if available

## Performance

### Training Speed

**Typical performance** (NVIDIA RTX 3090, batch_size=16):
- ~2-3 seconds per batch
- ~5-10 minutes per epoch (depending on dataset size)
- ~8-16 hours for 100 epochs

**CPU training**:
- ~10-20 seconds per batch
- ~30-60 minutes per epoch
- Not recommended for large datasets

### Memory Usage

| Configuration | GPU Memory | CPU Memory |
|---------------|------------|------------|
| batch_size=8 | ~2 GB | ~4 GB |
| batch_size=16 | ~4 GB | ~8 GB |
| batch_size=32 | ~8 GB | ~16 GB |

## Advanced Usage

### Custom Loss Weights

```python
# Edit train.py to customize loss combination
total_loss = (
    recon_loss + 
    kl_weight * kl_loss + 
    beat_weight * beat_loss +
    0.01 * custom_loss  # Add custom loss
)
```

### Different Warmup Schedules

```python
# Edit get_warmup_weights() function
# Current: Linear warmup
# Alternatives: Cosine, exponential, etc.
```

### Multiple GPUs

```bash
# Use DataParallel (automatic in PyTorch)
python train.py --data_dir ./weak_beat_music --batch_size 32
```

## Requirements

```bash
pip install torch torchvision librosa numpy tqdm tensorboard
```

## Project Structure

```
Final_model/
├── train.py                   # This training script
├── vae_model.py               # VAE model definition
├── beat_loss.py               # Beat loss function
├── music_dataset/
│   └── audio_dataset.py       # Dataset class
├── 1.filter_fma_weak_beat/
│   └── fma_filter.py          # Data filtering
├── checkpoints/               # Saved models
├── logs/                      # Tensorboard logs
└── weak_beat_music/           # Training data
```

---

**Ready to train!** Run `python train.py --help` for all options, or `python train.py` to start training with defaults.
