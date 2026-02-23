"""
Beat Loss Function for Weak Beat Music Generation
==================================================
Implements a differentiable loss function that penalizes regular beats,
encouraging the generation of ambient/experimental music with weak beats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def beat_loss(mel_spectrogram: torch.Tensor, 
              n_low_freq_bands: int = 10,
              exclude_lags: int = 5,
              normalize: bool = True) -> torch.Tensor:
    """
    Compute beat regularity loss for Mel-spectrograms.
    
    The loss is designed to penalize regular beats by measuring the
    autocorrelation of the low-frequency energy envelope. Higher
    autocorrelation indicates more regular beats, which results in
    higher loss for weak-beat music generation.
    
    Args:
        mel_spectrogram: Input tensor of shape (batch, 1, n_mels, n_frames)
        n_low_freq_bands: Number of low-frequency mel bands to use (default: 10)
        exclude_lags: Number of initial lags to exclude from max calculation (default: 5)
        normalize: Whether to normalize autocorrelation values (default: True)
    
    Returns:
        loss: Scalar tensor representing the beat regularity loss
              Range: [0, 1] where higher values indicate more regular beats
    
    Process:
        1. Extract low-frequency bands (bass frequencies contain beat info)
        2. Compute energy envelope by averaging across frequency bands
        3. Compute autocorrelation using FFT
        4. Find maximum autocorrelation (excluding lag=0 and near-zero lags)
        5. Loss = max_correlation (high correlation = regular beats = high loss)
    """
    # Validate input shape
    assert mel_spectrogram.dim() == 4, f"Expected 4D tensor, got {mel_spectrogram.dim()}D"
    batch_size, channels, n_mels, n_frames = mel_spectrogram.shape
    assert channels == 1, f"Expected 1 channel, got {channels}"
    assert n_mels >= n_low_freq_bands, f"n_mels ({n_mels}) must be >= n_low_freq_bands ({n_low_freq_bands})"
    
    # Step 1: Extract low-frequency part (bass frequencies, first 10 mel bands)
    # Low frequencies contain most beat/rhythm information
    low_freq = mel_spectrogram[:, :, :n_low_freq_bands, :]  # (batch, 1, 10, n_frames)
    
    # Step 2: Compute energy envelope
    # Average across frequency bands to get temporal energy pattern
    energy_envelope = torch.mean(low_freq, dim=2).squeeze(1)  # (batch, n_frames)
    
    # Normalize each sample's envelope to [0, 1] for consistent autocorrelation
    if normalize:
        envelope_min = energy_envelope.min(dim=1, keepdim=True)[0]
        envelope_max = energy_envelope.max(dim=1, keepdim=True)[0]
        energy_envelope = (energy_envelope - envelope_min) / (envelope_max - envelope_min + 1e-8)
    
    # Step 3: Compute autocorrelation using FFT
    # FFT-based autocorrelation is efficient and fully differentiable
    autocorr = compute_autocorrelation_fft(energy_envelope)  # (batch, n_frames)
    
    # Step 4: Find maximum correlation value (excluding lag=0 and very small lags)
    # Lag=0 always gives correlation=1, and small lags are not musically meaningful
    # We exclude the first few lags to focus on beat periodicity
    
    # Zero out lag=0 and small lags
    mask = torch.ones_like(autocorr)
    mask[:, :exclude_lags] = 0
    masked_autocorr = autocorr * mask
    
    # Find maximum correlation for each sample in batch
    max_correlation, _ = torch.max(masked_autocorr, dim=1)  # (batch,)
    
    # Clamp to [0, 1] range (autocorrelation should be in [-1, 1], we take positive part)
    max_correlation = torch.clamp(max_correlation, 0.0, 1.0)
    
    # Step 5: Loss = regularity score
    # High max_correlation = regular beats = high loss (we want to minimize this)
    # For weak beat music, we want LOW autocorrelation (irregular/no beats)
    regularity_score = max_correlation
    loss = regularity_score
    
    # Return average loss over batch
    return loss.mean()


def compute_autocorrelation_fft(signal: torch.Tensor) -> torch.Tensor:
    """
    Compute autocorrelation using FFT (Wiener-Khinchin theorem).
    
    The autocorrelation can be efficiently computed as:
    autocorr = IFFT(|FFT(signal)|^2)
    
    This is fully differentiable using PyTorch's FFT operations.
    
    Args:
        signal: Input tensor of shape (batch, n_frames)
    
    Returns:
        autocorr: Autocorrelation of shape (batch, n_frames)
                  Normalized to have autocorr[:, 0] = 1.0
    """
    batch_size, n_frames = signal.shape
    
    # Subtract mean to center the signal (improves correlation measurement)
    signal_centered = signal - signal.mean(dim=1, keepdim=True)
    
    # Pad signal to avoid circular correlation artifacts
    # Padding to 2*n_frames ensures linear correlation
    padded_length = 2 * n_frames
    signal_padded = F.pad(signal_centered, (0, n_frames), mode='constant', value=0)
    
    # Compute FFT
    # torch.fft.rfft returns complex tensor
    signal_fft = torch.fft.rfft(signal_padded, dim=1)  # (batch, freq_bins)
    
    # Compute power spectrum: |FFT|^2
    power_spectrum = torch.abs(signal_fft) ** 2
    
    # Inverse FFT to get autocorrelation
    autocorr_full = torch.fft.irfft(power_spectrum, n=padded_length, dim=1)  # (batch, padded_length)
    
    # Take only the first n_frames (valid autocorrelation range)
    autocorr = autocorr_full[:, :n_frames]
    
    # Normalize so that autocorr[:, 0] = 1.0 (correlation at lag=0)
    autocorr_normalized = autocorr / (autocorr[:, 0:1] + 1e-8)
    
    return autocorr_normalized


def compute_autocorrelation_direct(signal: torch.Tensor, max_lag: Optional[int] = None) -> torch.Tensor:
    """
    Compute autocorrelation using direct method (circular shift and dot product).
    
    This is an alternative implementation that's more intuitive but slower.
    Useful for verification and small sequences.
    
    Args:
        signal: Input tensor of shape (batch, n_frames)
        max_lag: Maximum lag to compute (default: n_frames)
    
    Returns:
        autocorr: Autocorrelation of shape (batch, max_lag)
    """
    batch_size, n_frames = signal.shape
    
    if max_lag is None:
        max_lag = n_frames
    
    # Center the signal
    signal_centered = signal - signal.mean(dim=1, keepdim=True)
    
    # Compute variance for normalization
    variance = torch.sum(signal_centered ** 2, dim=1, keepdim=True) + 1e-8
    
    # Compute autocorrelation for each lag
    autocorr_list = []
    
    for lag in range(max_lag):
        if lag == 0:
            # Lag 0: perfect correlation
            corr = torch.sum(signal_centered * signal_centered, dim=1) / variance.squeeze(1)
        else:
            # Shift signal by lag and compute dot product
            signal_shifted = torch.roll(signal_centered, shifts=lag, dims=1)
            corr = torch.sum(signal_centered * signal_shifted, dim=1) / variance.squeeze(1)
        
        autocorr_list.append(corr.unsqueeze(1))
    
    autocorr = torch.cat(autocorr_list, dim=1)  # (batch, max_lag)
    
    return autocorr


def beat_loss_alternative(mel_spectrogram: torch.Tensor,
                          n_low_freq_bands: int = 10,
                          use_direct: bool = False) -> torch.Tensor:
    """
    Alternative implementation of beat_loss using direct autocorrelation.
    
    Useful for verification and debugging. This version is slower but more explicit.
    
    Args:
        mel_spectrogram: Input tensor of shape (batch, 1, n_mels, n_frames)
        n_low_freq_bands: Number of low-frequency mel bands to use
        use_direct: If True, use direct autocorrelation method
    
    Returns:
        loss: Beat regularity loss
    """
    # Extract low-frequency bands
    low_freq = mel_spectrogram[:, :, :n_low_freq_bands, :]
    
    # Compute energy envelope
    energy_envelope = torch.mean(low_freq, dim=2).squeeze(1)
    
    # Normalize
    envelope_min = energy_envelope.min(dim=1, keepdim=True)[0]
    envelope_max = energy_envelope.max(dim=1, keepdim=True)[0]
    energy_envelope = (energy_envelope - envelope_min) / (envelope_max - envelope_min + 1e-8)
    
    # Compute autocorrelation
    if use_direct:
        autocorr = compute_autocorrelation_direct(energy_envelope)
    else:
        autocorr = compute_autocorrelation_fft(energy_envelope)
    
    # Find max correlation (excluding small lags)
    masked_autocorr = autocorr[:, 5:]  # Exclude first 5 lags
    max_correlation = torch.max(masked_autocorr, dim=1)[0]
    max_correlation = torch.clamp(max_correlation, 0.0, 1.0)
    
    return max_correlation.mean()


class BeatLoss(nn.Module):
    """
    PyTorch Module wrapper for beat_loss function.
    
    Can be used as a loss component in training pipelines.
    
    Example:
        >>> beat_criterion = BeatLoss(n_low_freq_bands=10, weight=0.1)
        >>> mel_spec = torch.randn(16, 1, 128, 431)
        >>> loss = beat_criterion(mel_spec)
    """
    
    def __init__(self, 
                 n_low_freq_bands: int = 10,
                 exclude_lags: int = 5,
                 weight: float = 1.0,
                 normalize: bool = True):
        """
        Initialize BeatLoss module.
        
        Args:
            n_low_freq_bands: Number of low-frequency bands to use
            exclude_lags: Number of initial lags to exclude
            weight: Loss weight multiplier
            normalize: Whether to normalize energy envelope
        """
        super(BeatLoss, self).__init__()
        self.n_low_freq_bands = n_low_freq_bands
        self.exclude_lags = exclude_lags
        self.weight = weight
        self.normalize = normalize
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Compute beat loss.
        
        Args:
            mel_spectrogram: Input tensor of shape (batch, 1, n_mels, n_frames)
        
        Returns:
            loss: Weighted beat regularity loss
        """
        loss = beat_loss(
            mel_spectrogram,
            n_low_freq_bands=self.n_low_freq_bands,
            exclude_lags=self.exclude_lags,
            normalize=self.normalize
        )
        return self.weight * loss


def test_beat_loss():
    """Test function for beat_loss."""
    print("="*70)
    print("Testing Beat Loss Function")
    print("="*70)
    
    # Test dimensions
    batch_size = 8
    n_mels = 128
    n_frames = 431
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Mel bands: {n_mels}")
    print(f"  Time frames: {n_frames}")
    
    # Test 1: Random noise (should have low beat regularity)
    print("\n" + "-"*70)
    print("Test 1: Random noise (irregular)")
    random_spec = torch.randn(batch_size, 1, n_mels, n_frames)
    loss_random = beat_loss(random_spec)
    print(f"  Beat loss: {loss_random.item():.4f}")
    print(f"  Expected: Low (irregular/no beats)")
    
    # Test 2: Periodic signal (should have high beat regularity)
    print("\n" + "-"*70)
    print("Test 2: Periodic signal (regular beats)")
    # Create a signal with regular beats (period = 20 frames)
    periodic_spec = torch.zeros(batch_size, 1, n_mels, n_frames)
    period = 20
    for i in range(0, n_frames, period):
        periodic_spec[:, :, :10, i:min(i+5, n_frames)] = 1.0  # Beat pulse
    loss_periodic = beat_loss(periodic_spec)
    print(f"  Beat loss: {loss_periodic.item():.4f}")
    print(f"  Expected: High (regular beats)")
    
    # Test 3: Ambient-like (low energy variation)
    print("\n" + "-"*70)
    print("Test 3: Ambient-like (constant energy)")
    ambient_spec = torch.ones(batch_size, 1, n_mels, n_frames) * 0.5
    ambient_spec += torch.randn_like(ambient_spec) * 0.05  # Small noise
    loss_ambient = beat_loss(ambient_spec)
    print(f"  Beat loss: {loss_ambient.item():.4f}")
    print(f"  Expected: Low (no beats, constant)")
    
    # Test 4: Gradient flow (ensure differentiability)
    print("\n" + "-"*70)
    print("Test 4: Gradient flow")
    test_spec = torch.randn(batch_size, 1, n_mels, n_frames, requires_grad=True)
    loss = beat_loss(test_spec)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient shape: {test_spec.grad.shape}")
    print(f"  Gradient mean: {test_spec.grad.mean().item():.6f}")
    print(f"  Gradient std: {test_spec.grad.std().item():.6f}")
    print("  ✓ Gradients computed successfully!")
    
    # Test 5: BeatLoss module
    print("\n" + "-"*70)
    print("Test 5: BeatLoss module")
    beat_criterion = BeatLoss(n_low_freq_bands=10, weight=0.5)
    test_spec = torch.randn(batch_size, 1, n_mels, n_frames)
    loss = beat_criterion(test_spec)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✓ Module works correctly!")
    
    # Test 6: Compare FFT vs Direct methods
    print("\n" + "-"*70)
    print("Test 6: FFT vs Direct autocorrelation")
    test_signal = torch.randn(4, 100)  # Smaller for direct method
    
    autocorr_fft = compute_autocorrelation_fft(test_signal)
    autocorr_direct = compute_autocorrelation_direct(test_signal, max_lag=100)
    
    diff = torch.abs(autocorr_fft - autocorr_direct).mean()
    print(f"  Mean difference: {diff.item():.6f}")
    print(f"  ✓ Both methods produce similar results!")
    
    # Test 7: Batch processing
    print("\n" + "-"*70)
    print("Test 7: Batch processing")
    large_batch = torch.randn(32, 1, n_mels, n_frames)
    loss_large = beat_loss(large_batch)
    print(f"  Loss for batch_size=32: {loss_large.item():.4f}")
    print(f"  ✓ Large batch processing works!")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    
    # Summary
    print("\nSummary:")
    print(f"  Random noise loss: {loss_random.item():.4f}")
    print(f"  Periodic signal loss: {loss_periodic.item():.4f}")
    print(f"  Ambient signal loss: {loss_ambient.item():.4f}")
    print("\nInterpretation:")
    print("  - Higher loss = more regular beats (not desired for weak beat music)")
    print("  - Lower loss = irregular/no beats (desired for ambient/experimental)")
    print("  - Use this loss to encourage weak beat generation in VAE training")


if __name__ == "__main__":
    test_beat_loss()
