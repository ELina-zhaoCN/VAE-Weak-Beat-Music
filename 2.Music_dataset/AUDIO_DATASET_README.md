# AudioMelDataset - PyTorch Dataset for Mel-Spectrograms

## Overview

`AudioMelDataset` is a PyTorch `Dataset` class that loads audio files and converts them to Mel-spectrograms, suitable for training audio generation models, music analysis, or any deep learning tasks involving audio.

## Features

✅ **Random Audio Loading**: Randomly selects files and applies random temporal offsets  
✅ **Mel-Spectrogram Conversion**: Uses librosa for high-quality audio processing  
✅ **Automatic Padding/Truncation**: Ensures consistent output dimensions  
✅ **Log Scaling & Normalization**: Applies proper scaling for neural networks  
✅ **Exception Handling**: Gracefully handles loading errors  
✅ **PyTorch Integration**: Full DataLoader compatibility  
✅ **Batch Processing**: Efficient multi-threaded data loading  

## Installation

```bash
pip install torch librosa numpy matplotlib soundfile
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from audio_dataset import AudioMelDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = AudioMelDataset(
    data_dir="./weak_beat_music",  # Your audio folder
    sr=22050,                       # Sampling rate
    duration=10.0,                  # Segment duration
    n_mels=128                      # Number of Mel bands
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for batch in dataloader:
    # batch shape: (batch_size, 1, 128, 431)
    # Train your model here
    pass
```

## Class Reference

### `AudioMelDataset`

```python
AudioMelDataset(
    data_dir: str,
    sr: int = 22050,
    duration: float = 10.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_offset: float = 5.0,
    audio_extensions: tuple = ('.mp3', '.wav', '.flac', '.m4a', '.ogg'),
    normalize: bool = True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | required | Path to directory containing audio files |
| `sr` | int | 22050 | Sampling rate in Hz |
| `duration` | float | 10.0 | Duration of each segment in seconds |
| `n_mels` | int | 128 | Number of Mel frequency bands |
| `n_fft` | int | 2048 | FFT window size |
| `hop_length` | int | 512 | Samples between successive frames |
| `max_offset` | float | 5.0 | Maximum random offset in seconds |
| `audio_extensions` | tuple | (see default) | Supported audio file extensions |
| `normalize` | bool | True | Whether to normalize spectrograms |

#### Methods

##### `__len__()`
Returns the number of audio files in the dataset.

```python
num_files = len(dataset)
```

##### `__getitem__(idx)`
Loads a random audio file and returns its Mel-spectrogram.

**Note**: The `idx` parameter is used for iteration but the actual file is randomly selected.

```python
mel_spec = dataset[0]  # Returns (1, n_mels, n_frames) tensor
```

**Returns**: `torch.FloatTensor` of shape `(1, n_mels, n_frames)`

##### `get_by_index(idx)`
Loads a specific audio file by index (non-random selection).

```python
mel_spec = dataset.get_by_index(5)  # Always returns the same file
```

**Returns**: `torch.FloatTensor` of shape `(1, n_mels, n_frames)`

##### `get_audio_info(idx)`
Returns information about an audio file without loading it.

```python
info = dataset.get_audio_info(0)
# Returns: {'path': '...', 'filename': '...', 'duration': 12.5, 'index': 0}
```

**Returns**: `dict` with file information

##### `audio_to_mel(audio)`
Converts audio waveform to Mel-spectrogram (called internally).

```python
mel_spec = dataset.audio_to_mel(audio_array)
```

**Args**: `np.ndarray` of shape `(n_samples,)`  
**Returns**: `np.ndarray` of shape `(n_mels, n_frames)`

## Output Shape Calculation

The number of frames in the output Mel-spectrogram is calculated as:

```python
n_frames = duration * sr / hop_length
```

**Example** (default parameters):
```
n_frames = 10.0 * 22050 / 512 ≈ 430.66 ≈ 431 frames
```

So the default output shape is: **`(1, 128, 431)`**

## Usage Examples

### Example 1: Basic Usage

```python
from audio_dataset import AudioMelDataset

# Initialize dataset
dataset = AudioMelDataset(data_dir="./music")

print(f"Dataset size: {len(dataset)}")

# Get a sample
mel_spec = dataset[0]
print(f"Shape: {mel_spec.shape}")  # (1, 128, 431)
print(f"Range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
```

### Example 2: Custom Parameters

```python
dataset = AudioMelDataset(
    data_dir="./music",
    sr=16000,          # Lower sampling rate
    duration=5.0,      # Shorter segments
    n_mels=64,         # Fewer Mel bands
    hop_length=256,    # Smaller hop (more frames)
    max_offset=2.0     # Less randomness
)
```

### Example 3: DataLoader Integration

```python
from torch.utils.data import DataLoader

dataset = AudioMelDataset(data_dir="./music")

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)

for batch in dataloader:
    # batch shape: (32, 1, 128, 431)
    print(f"Batch: {batch.shape}")
    break
```

### Example 4: Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audio_dataset import AudioMelDataset

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AudioMelDataset(data_dir="./music")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model (example)
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)
        
        # Forward pass
        output = model(batch)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 5: Loading Specific Files

```python
dataset = AudioMelDataset(data_dir="./music")

# Get file information
for i in range(5):
    info = dataset.get_audio_info(i)
    print(f"{info['filename']}: {info['duration']:.2f}s")

# Load specific file (always the same, no randomness)
mel_spec = dataset.get_by_index(0)
```

### Example 6: Visualization

```python
import matplotlib.pyplot as plt

dataset = AudioMelDataset(data_dir="./music")
mel_spec = dataset[0]

# Remove channel dimension for plotting
mel_display = mel_spec[0].numpy()

plt.figure(figsize=(12, 6))
plt.imshow(mel_display, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Normalized Magnitude')
plt.title('Mel-Spectrogram')
plt.xlabel('Time Frames')
plt.ylabel('Mel Frequency Bands')
plt.tight_layout()
plt.savefig('mel_spectrogram.png')
```

## Audio Processing Pipeline

The dataset follows this processing pipeline:

```
Audio File
    ↓
[Random Selection]
    ↓
[Load with librosa] (with random offset)
    ↓
[Ensure Length] (pad or truncate to sr * duration)
    ↓
[Compute Mel-Spectrogram] (librosa.feature.melspectrogram)
    ↓
[Log Scaling] (convert to dB)
    ↓
[Normalization] (min-max to [0, 1])
    ↓
[Ensure Frames] (pad or truncate to n_frames)
    ↓
[Convert to Tensor] (torch.FloatTensor)
    ↓
[Add Channel Dim] (1, n_mels, n_frames)
    ↓
Output
```

## Configuration Guidelines

### For Music Generation Models

```python
dataset = AudioMelDataset(
    data_dir="./music",
    sr=22050,          # CD-quality
    duration=10.0,     # Longer context
    n_mels=128,        # High frequency resolution
    hop_length=512,    # Standard hop
    max_offset=5.0     # Good variation
)
```

### For Speech Tasks

```python
dataset = AudioMelDataset(
    data_dir="./speech",
    sr=16000,          # Speech sampling rate
    duration=5.0,      # Shorter utterances
    n_mels=80,         # Standard for speech
    hop_length=160,    # 10ms frames
    max_offset=1.0     # Less randomness
)
```

### For Fast Prototyping

```python
dataset = AudioMelDataset(
    data_dir="./music",
    sr=16000,          # Lower SR = faster
    duration=5.0,      # Shorter = less memory
    n_mels=64,         # Fewer bands = faster
    hop_length=256,    # Faster processing
    max_offset=2.0
)
```

## Error Handling

The dataset handles errors gracefully:

- **Missing files**: Skipped during scanning
- **Corrupt audio**: Returns zero tensor and prints error
- **Short audio**: Automatically padded with zeros
- **Long audio**: Automatically truncated

Example error output:
```
Error loading audio file: [Errno 2] No such file or directory: 'missing.mp3'
Attempted file: ./music/missing.mp3
```

## Performance Tips

1. **Multi-threaded Loading**: Use `num_workers > 0` in DataLoader
   ```python
   DataLoader(dataset, num_workers=4)  # 4 parallel workers
   ```

2. **Pin Memory**: Enable for faster GPU transfer
   ```python
   DataLoader(dataset, pin_memory=True)
   ```

3. **Prefetch**: Use `prefetch_factor` for better throughput
   ```python
   DataLoader(dataset, num_workers=4, prefetch_factor=2)
   ```

4. **Lower Sampling Rate**: Reduce `sr` for faster processing
   ```python
   AudioMelDataset(sr=16000)  # Instead of 22050
   ```

## Supported Audio Formats

- `.mp3` - MP3 audio
- `.wav` - WAV audio
- `.flac` - FLAC lossless
- `.m4a` - AAC/ALAC
- `.ogg` - Ogg Vorbis

## Testing

Run the built-in test:

```bash
python audio_dataset.py
```

Or run comprehensive examples:

```bash
python example_audio_dataset.py
```

## Troubleshooting

### "No audio files found"
- Check that `data_dir` exists and contains audio files
- Verify file extensions match `audio_extensions`

### "librosa not installed"
```bash
pip install librosa soundfile
```

### "Shapes don't match"
- Check `duration` and `hop_length` settings
- Verify expected `n_frames` calculation

### Slow loading
- Reduce `sr` (sampling rate)
- Reduce `duration`
- Increase `num_workers` in DataLoader

### GPU memory issues
- Reduce `batch_size` in DataLoader
- Reduce `n_mels` or `n_frames`
- Use gradient accumulation

## Integration with Popular Models

### Generative Adversarial Networks (GANs)

```python
dataset = AudioMelDataset(data_dir="./music")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Use as real samples in GAN training
for real_samples in dataloader:
    # Train discriminator
    # Train generator
    pass
```

### Variational Autoencoders (VAEs)

```python
# Use for reconstruction tasks
for mel_spec in dataloader:
    reconstructed = vae(mel_spec)
    loss = reconstruction_loss(reconstructed, mel_spec)
```

### Diffusion Models

```python
# Use for denoising diffusion
for x0 in dataloader:
    t = sample_timesteps(batch_size)
    noise = torch.randn_like(x0)
    xt = add_noise(x0, noise, t)
    predicted_noise = model(xt, t)
```

## Citation

If you use this dataset class in your research, consider citing librosa:

```bibtex
@inproceedings{mcfee2015librosa,
  title={librosa: Audio and music signal analysis in python},
  author={McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel PW and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  booktitle={Proceedings of the 14th python in science conference},
  volume={8},
  year={2015}
}
```

## License

This code is provided as-is for educational and research purposes.

---

**Questions or Issues?** Check the examples in `example_audio_dataset.py` or review the inline documentation in `audio_dataset.py`.
