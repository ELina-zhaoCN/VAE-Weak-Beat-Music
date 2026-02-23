# VAE for Weak-Beat Music Generation

A Variational Autoencoder (VAE) model that learns to generate rhythmic beats from weak-beat music (Ambient, Drone, Experimental genres).

## 🎯 Project Overview

This project implements a novel VAE architecture with a **beat regularization loss** that:
- Takes weak-beat music (e.g., Ambient, Drone) as input
- Learns to generate music with clear, regular beats
- Uses autocorrelation-based loss to measure beat regularity

**Key Innovation**: The model "sees high loss" for weak beats and automatically adjusts through backpropagation to generate more regular rhythmic patterns.

## 📊 Project Statistics

- **Code**: ~3,067 lines across 6 core modules
- **Dataset**: 2,272 weak-beat tracks filtered from FMA Medium
- **Training**: 100 epochs with progressive warm-up strategy
- **Test Results**: 5 samples with comprehensive evaluation

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and filter FMA dataset
cd 1.Filter_fma_weak_beat
python fma_filter.py --download-info
# (follow download instructions)
python fma_filter.py --filter --audio-dir ../fma_data/fma_medium
cd ..

# 3. Train the model
python train.py --data_dir ./weak_beat_music --epochs 100

# 4. Evaluate results
python evaluate.py --checkpoint ./checkpoints/best_model.pt --data_dir ./weak_beat_music
```

## 📁 Project Structure

```
Final_model/
├── 1.Filter_fma_weak_beat/    # Data filtering from FMA
├── 2.Music_dataset/            # PyTorch Dataset implementation
├── 3.Model/                    # VAE model architecture
├── 4.Beat_losses/              # Beat loss function
├── 5.Tran_Script/              # Training scripts
├── 6.Test_Script/              # Evaluation scripts
├── train.py                    # Main training script
├── vae_model.py                # VAE model definition
├── beat_loss.py                # Beat loss implementation
├── evaluate.py                 # Evaluation script
└── requirements.txt            # Python dependencies
```

## 🧠 Model Architecture

**Input**: Mel-spectrogram (1, 128, 431)  
**Latent Space**: 128 dimensions  
**Output**: Reconstructed Mel-spectrogram (1, 128, 431)

### Encoder
- 3 convolutional layers with BatchNorm and ReLU
- Progressive downsampling (128×431 → 16×53)
- Linear layers to 128D latent space (mu and logvar)

### Decoder
- Linear expansion from 128D latent vector
- 3 transposed convolutional layers
- Upsampling back to 128×431

## 🎵 Loss Function

```python
Total Loss = Reconstruction Loss + β × KL Divergence + γ × Beat Loss
```

### Beat Loss (Core Innovation)
1. Extract low-frequency energy (first 10 Mel bands)
2. Compute autocorrelation using FFT
3. Find maximum correlation (beat regularity score)
4. Loss = max_correlation

**Key Mechanism**:
- Weak beats → Low autocorrelation → Low loss
- Regular beats → High autocorrelation → High loss
- Model learns to **increase** beat regularity to minimize total loss!

### Progressive Warm-up
- First 20 epochs: β linearly increases 0→1.0, γ increases 0→0.5
- After epoch 20: Fixed at β=1.0, γ=0.5

## 📈 Training Results

**Dataset Split**:
- Training: 1,818 tracks (80%)
- Validation: 454 tracks (20%)

**Performance** (5 test samples):
- Average MSE: 0.0246 ± 0.0069
- BPM difference: 25.0 ± 15.5
- Beat regularity: 0.55 ± 0.20

## 🔧 Key Features

- **Data Filtering**: Automatic filtering of weak-beat genres from FMA
- **Custom Dataset**: PyTorch Dataset for Mel-spectrogram generation
- **Beat Loss**: Differentiable autocorrelation-based loss
- **Warm-up Strategy**: Progressive loss weight scheduling
- **Comprehensive Evaluation**: BPM detection, autocorrelation analysis
- **Visualization**: Waveform, spectrogram, and beat regularity plots

## 📝 Documentation

- `汇报提纲.md` - Project presentation outline (Chinese)
- `项目报告.md` - Full project report (Chinese)
- `README_*.md` - Component-specific documentation
- `PROJECT_README.md` - Detailed project overview

## 🛠️ Requirements

```
torch>=1.9.0
librosa>=0.9.0
numpy>=1.20.0
matplotlib>=3.3.0
tqdm>=4.60.0
soundfile>=0.10.0
pandas>=1.3.0
```

## 📊 Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--latent_dim` | 128 | Latent space dimension |
| `--kl_weight` | 1.0 | β (KL weight after warmup) |
| `--beat_weight` | 0.5 | γ (Beat weight after warmup) |
| `--warmup_epochs` | 20 | Warmup duration |

## 🎓 Course Information

- **Course**: 513 Managing Data And Signal Processing
- **Institution**: University of Washington
- **Author**: Elian Zhao
- **Date**: February 2026

## 📄 License

This project is for educational purposes.

## 🔗 References

- FMA Dataset: [mdeff/fma](https://github.com/mdeff/fma)
- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Librosa: Audio and music signal analysis library

---

**Note**: This project demonstrates how machine learning models can "see high loss and adjust themselves" through backpropagation to generate desired outputs - in this case, transforming weak-beat music into rhythmic, beat-driven music.
