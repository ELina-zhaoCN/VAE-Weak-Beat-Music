"""
Beat Loss v2 - FIXED Direction
================================
Key fix: loss = 1 - regularity_score
  - Minimizing this loss → model learns to OUTPUT music with STRONGER beats
  - v1 bug: loss = regularity_score → model learned to OUTPUT WEAKER beats (wrong!)

Additional fix: frequency-split energy extraction for cleaner beat signal.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def beat_loss_v2(
    mel_spectrogram: torch.Tensor,
    n_low_freq_bands: int = 16,
    bpm_min: int = 60,
    bpm_max: int = 240,
    sr: int = 22050,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Beat regularity loss v2 (FIXED direction).

    loss = 1 - max_autocorrelation
      → minimizing loss forces decoder to INCREASE beat regularity in output.

    Args:
        mel_spectrogram : (batch, 1, n_mels, n_frames)
        n_low_freq_bands: how many low-freq mel bands to use (default 16)
        bpm_min / bpm_max: BPM search range for meaningful lag selection
        sr, hop_length   : audio parameters to convert BPM → frame lags

    Returns:
        scalar loss ∈ [0, 1]  (0 = perfectly regular beat, 1 = no beat)
    """
    assert mel_spectrogram.dim() == 4, "Expected (B, 1, n_mels, n_frames)"
    batch, _, n_mels, n_frames = mel_spectrogram.shape

    # ── Step 1: extract low-freq energy envelope ──────────────────────────
    low_freq = mel_spectrogram[:, 0, :n_low_freq_bands, :]   # (B, bands, T)
    energy   = low_freq.mean(dim=1)                           # (B, T)

    # ── Step 2: normalize each sample to zero-mean, unit std ──────────────
    mu  = energy.mean(dim=1, keepdim=True)
    std = energy.std(dim=1, keepdim=True) + 1e-8
    energy_norm = (energy - mu) / std                         # (B, T)

    # ── Step 3: FFT-based autocorrelation ─────────────────────────────────
    padded = F.pad(energy_norm, (0, n_frames))                # (B, 2T)
    spec   = torch.fft.rfft(padded, dim=1)
    power  = torch.abs(spec) ** 2
    acf    = torch.fft.irfft(power, n=2 * n_frames, dim=1)[:, :n_frames]
    acf    = acf / (acf[:, 0:1] + 1e-8)                      # normalize lag-0=1

    # ── Step 4: find max autocorrelation in musically meaningful lag range ─
    lag_min = max(1, int(round(sr / (hop_length * bpm_max / 60))))
    lag_max = min(n_frames - 1, int(round(sr / (hop_length * bpm_min / 60))))
    lag_min = min(lag_min, lag_max - 1)                       # safety guard

    # zero out lags outside the BPM range so max() picks correctly
    mask = torch.zeros_like(acf)
    mask[:, lag_min:lag_max] = 1.0
    masked_acf = acf * mask

    regularity, _ = masked_acf.max(dim=1)                    # (B,)
    regularity = torch.clamp(regularity, 0.0, 1.0)

    # ── Step 5: FIXED loss direction ──────────────────────────────────────
    # v1 bug:  loss = regularity        → minimise = weaker beats  ❌
    # v2 fix:  loss = 1 - regularity    → minimise = stronger beats ✅
    loss = 1.0 - regularity

    return loss.mean()


def compute_regularity_score(mel_tensor: torch.Tensor,
                              n_low_freq_bands: int = 16,
                              bpm_min: int = 60,
                              bpm_max: int = 240,
                              sr: int = 22050,
                              hop_length: int = 512) -> float:
    """
    Returns beat regularity score in [0, 1].
    1.0 = perfectly regular, 0.0 = no detectable beat.
    Convenience wrapper for evaluation / inference.
    """
    with torch.no_grad():
        loss = beat_loss_v2(mel_tensor, n_low_freq_bands,
                            bpm_min, bpm_max, sr, hop_length)
    return float(1.0 - loss.item())


if __name__ == "__main__":
    # Quick sanity check
    B, T = 4, 431

    # Regular beat signal (period ~22 frames ≈ 120 BPM at 22050/512)
    regular = torch.zeros(B, 1, 128, T)
    for t in range(0, T, 22):
        regular[:, :, :16, t:t+3] = 1.0
    l_reg = beat_loss_v2(regular)
    s_reg = compute_regularity_score(regular)
    print(f"Regular beat  → loss={l_reg:.4f}, regularity={s_reg:.4f}  (expect low loss)")

    # Ambient / no beat
    ambient = torch.randn(B, 1, 128, T) * 0.1
    l_amb = beat_loss_v2(ambient)
    s_amb = compute_regularity_score(ambient)
    print(f"Ambient/noise → loss={l_amb:.4f}, regularity={s_amb:.4f}  (expect high loss)")

    assert l_reg < l_amb, "BUG: regular beat should have LOWER loss than ambient"
    print("✓ Loss direction correct: regular < ambient")
