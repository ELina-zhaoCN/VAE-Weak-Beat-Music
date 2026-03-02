# Convolutional VAE for Mel-Spectrograms

PyTorch implementation of a Variational Autoencoder (VAE) for Mel-spectrogram generation.

## Architecture

**Input**: `(batch_size, 1, 128, 431)`  
**Latent Dimension**: `128`  
**Output**: `(batch_size, 1, 128, 431)`

### Encoder

```
Input (1, 128, 431)
  ↓
Conv2d(1→32, k=4, s=2, p=1) + BatchNorm + ReLU → (32, 64, 215)
  ↓
Conv2d(32→64, k=4, s=2, p=1) + BatchNorm + ReLU → (64, 32, 107)
  ↓
Conv2d(64→128, k=4, s=2, p=1) + BatchNorm + ReLU → (128, 16, 53)
  ↓
Flatten → (108,544)
  ↓
Linear → mu (128), logvar (128)
```

**Flattened dimension**: `128 × 16 × 53 = 108,544`

### Decoder

```
Input: z (128)
  ↓
Linear(128 → 108,544)
  ↓
Reshape → (128, 16, 53)
  ↓
ConvTranspose2d(128→64, k=4, s=2, p=1) + BatchNorm + ReLU → (64, 32, 106)
  ↓
ConvTranspose2d(64→32, k=4, s=2, p=1) + BatchNorm + ReLU → (32, 64, 212)
  ↓
ConvTranspose2d(32→1, k=4, s=2, p=1) → (1, 128, 424)
  ↓
Conv2d(1→1, k=3, s=1, p=1) → (1, 128, 424)
  ↓
Padding (424→431) → (1, 128, 431)
```

## Quick Start

### 1. Test Model

```bash
python vae_model.py
```

### 2. Train Model

```bash
python train_vae.py
```

## Usage

### Basic Usage

```python
from vae_model import MelSpectrogramVAE, vae_loss
import torch

# Create model
model = MelSpectrogramVAE(latent_dim=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
input_batch = torch.randn(16, 1, 128, 431).to(device)
reconstruction, mu, logvar = model(input_batch)

# Compute loss
loss, loss_dict = vae_loss(reconstruction, input_batch, mu, logvar)
print(f"Loss: {loss_dict['total_loss']:.4f}")
```

### Generate Samples

```python
# Generate new spectrograms
samples = model.sample(num_samples=10, device=device)
print(f"Generated: {samples.shape}")  # (10, 1, 128, 431)
```

### Training

```python
from train_vae import VAETrainer

trainer = VAETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    kl_weight=1.0
)

trainer.train(num_epochs=50, save_interval=5)
```

## Model Methods

- `encode(x)` → `(mu, logvar)`: Encode input to latent parameters
- `reparameterize(mu, logvar)` → `z`: Sample from latent distribution
- `decode(z)` → `reconstruction`: Decode latent vector to spectrogram
- `forward(x)` → `(reconstruction, mu, logvar)`: Complete forward pass
- `sample(num_samples, device)` → `samples`: Generate new samples
- `reconstruct(x)` → `reconstruction`: Deterministic reconstruction (uses mu only)

## Loss Function

```
Total Loss = Reconstruction Loss + β × KL Divergence
```

- **Reconstruction Loss**: MSE between input and reconstruction
- **KL Divergence**: Regularizes latent space to standard normal distribution
- **β (kl_weight)**: Controls trade-off (default: 1.0)

## Training Features

✓ Automatic checkpoint saving  
✓ Best model selection  
✓ Learning rate scheduling  
✓ Gradient clipping  
✓ Training history logging  
✓ Sample generation  
✓ Training curve plotting  

## Output Files

```
checkpoints/
├── best_model.pt              # Best validation loss
├── final_model.pt             # Final model
├── checkpoint_epoch_N.pt      # Periodic checkpoints
├── training_history.json      # Loss history
├── training_curves.png        # Training plots
└── samples_epoch_N.png        # Generated samples
```

## Complete Pipeline

```bash
# 1. Filter audio data
cd filter_fma_weak_beat
python fma_filter.py --filter --audio-dir ./fma_data/fma_medium
cd ..

# 2. Train VAE
python train_vae.py

# 3. Check results
ls checkpoints/
```

## Requirements

```bash
pip install torch torchvision librosa numpy matplotlib
```

## Dimension Calculations

| Layer | Input (H×W) | Output (H×W) | Calculation |
|-------|-------------|--------------|-------------|
| Conv1 | 128×431 | 64×215 | (128+2-4)/2+1=64, (431+2-4)/2+1=215 |
| Conv2 | 64×215 | 32×107 | (64+2-4)/2+1=32, (215+2-4)/2+1=107 |
| Conv3 | 32×107 | 16×53 | (32+2-4)/2+1=16, (107+2-4)/2+1=53 |
| TransConv1 | 16×53 | 32×106 | (16-1)×2-2+4=32, (53-1)×2-2+4=106 |
| TransConv2 | 32×106 | 64×212 | (32-1)×2-2+4=64, (106-1)×2-2+4=212 |
| TransConv3 | 64×212 | 128×424 | (64-1)×2-2+4=128, (212-1)×2-2+4=424 |

## Hyperparameters

```python
config = {
    'latent_dim': 128,      # Latent space dimension
    'batch_size': 16,       # Batch size
    'learning_rate': 1e-4,  # Adam learning rate
    'kl_weight': 1.0,       # KL divergence weight
    'num_epochs': 50        # Training epochs
}
```

## Applications

- Music generation
- Audio interpolation
- Style transfer
- Audio denoising
- Data augmentation

## Model Statistics

- **Parameters**: ~50-60M
- **Memory Usage**: ~2-4 GB (batch_size=16)
- **Training Speed**: Depends on GPU

---

**Ready to use!** Run `python vae_model.py` to test, then `python train_vae.py` to train.
