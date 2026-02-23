#!/usr/bin/env python3
"""
Example script demonstrating how to use the AudioMelDataset class.
Shows various usage patterns including basic loading, DataLoader integration,
and batch processing.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the dataset class
from audio_dataset import AudioMelDataset


def example_1_basic_usage():
    """Example 1: Basic dataset usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Dataset Usage")
    print("="*70)
    
    # Initialize dataset
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",  # Your audio folder
        sr=22050,                       # 22.05 kHz sampling rate
        duration=10.0,                  # 10 seconds per segment
        n_mels=128,                     # 128 Mel bands
        hop_length=512                  # Hop length for STFT
    )
    
    print(f"Dataset size: {len(dataset)} audio files")
    print(f"Expected output shape: (1, 128, ~431)")
    
    # Get a random sample
    mel_spec = dataset[0]  # Randomly selects a file
    print(f"\nSample shape: {mel_spec.shape}")
    print(f"Data type: {mel_spec.dtype}")
    print(f"Device: {mel_spec.device}")
    print(f"Value range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    
    return dataset


def example_2_dataloader():
    """Example 2: Using with PyTorch DataLoader."""
    print("\n" + "="*70)
    print("EXAMPLE 2: DataLoader Integration")
    print("="*70)
    
    # Create dataset
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",
        sr=22050,
        duration=10.0,
        n_mels=128
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    print(f"DataLoader created with batch_size=8")
    print(f"Total batches: {len(dataloader)}")
    
    # Iterate through a few batches
    print("\nSampling 3 batches:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i+1}: shape={batch.shape}, "
              f"range=[{batch.min():.4f}, {batch.max():.4f}]")
    
    return dataloader


def example_3_specific_files():
    """Example 3: Loading specific files by index."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Loading Specific Files")
    print("="*70)
    
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",
        sr=22050,
        duration=10.0,
        n_mels=128
    )
    
    # Get info about first 5 files
    print(f"\nFirst 5 audio files:")
    for i in range(min(5, len(dataset))):
        info = dataset.get_audio_info(i)
        print(f"  [{i}] {info['filename']}")
        if info.get('duration'):
            print(f"      Duration: {info['duration']:.2f}s")
    
    # Load a specific file
    if len(dataset) > 0:
        print(f"\nLoading file at index 0 (non-random):")
        mel_spec = dataset.get_by_index(0)
        print(f"  Shape: {mel_spec.shape}")


def example_4_custom_parameters():
    """Example 4: Using custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Parameters")
    print("="*70)
    
    # Create dataset with custom parameters
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",
        sr=16000,              # Lower sampling rate
        duration=5.0,          # Shorter segments
        n_mels=64,             # Fewer Mel bands
        hop_length=256,        # Smaller hop length (more frames)
        max_offset=2.0,        # Smaller random offset
        normalize=True         # Enable normalization
    )
    
    print(f"Custom configuration:")
    print(f"  Sampling rate: {dataset.sr} Hz")
    print(f"  Duration: {dataset.duration}s")
    print(f"  Mel bands: {dataset.n_mels}")
    print(f"  Expected frames: {dataset.n_frames}")
    print(f"  Max offset: {dataset.max_offset}s")
    
    mel_spec = dataset[0]
    print(f"\nOutput shape: {mel_spec.shape}")


def example_5_visualization():
    """Example 5: Visualizing Mel-spectrograms."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Visualization")
    print("="*70)
    
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",
        sr=22050,
        duration=10.0,
        n_mels=128
    )
    
    # Create figure with multiple samples
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print("Generating 4 random samples...")
    
    for i in range(4):
        mel_spec = dataset[i]  # Random sample
        mel_display = mel_spec[0].numpy()  # Remove channel dimension
        
        # Plot
        im = axes[i].imshow(
            mel_display,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Time Frames')
        axes[i].set_ylabel('Mel Frequency Bands')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    output_path = 'mel_spectrogram_batch.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def example_6_training_loop():
    """Example 6: Simple training loop demonstration."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Training Loop Pattern")
    print("="*70)
    
    # Setup
    dataset = AudioMelDataset(
        data_dir="./weak_beat_music",
        sr=22050,
        duration=10.0,
        n_mels=128
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for debugging, increase for training
    )
    
    # Dummy model (just for demonstration)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    
    # Simulate training loop
    print("\nSimulating 3 training iterations:")
    model.train()
    
    for epoch in range(3):
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Just show first 2 batches
                break
            
            # Move to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: "
                  f"input {batch.shape} -> output {output.shape}")
    
    print("\n✓ Training loop pattern demonstrated successfully!")


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              AudioMelDataset - Usage Examples                        ║
║                                                                      ║
║   Demonstrates how to use the PyTorch Dataset for Mel-spectrograms  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if dataset directory exists
    data_dir = Path("./weak_beat_music")
    if not data_dir.exists():
        print("⚠ Warning: Dataset directory './weak_beat_music' not found.")
        print("\nPlease run one of the following first:")
        print("  1. python fma_filter.py --filter --audio-dir ./fma_data/fma_medium")
        print("  2. python fma_filter.py --filter-local ~/Music --keywords ambient drone")
        print("\nOr modify the data_dir in the examples to point to your audio folder.")
        return
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_dataloader()
        example_3_specific_files()
        example_4_custom_parameters()
        example_5_visualization()
        example_6_training_loop()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Modify parameters in audio_dataset.py for your use case")
        print("  2. Integrate with your training pipeline")
        print("  3. Experiment with different audio augmentations")
        print("  4. Train your audio generation or analysis model")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples cancelled by user.")
        sys.exit(0)
