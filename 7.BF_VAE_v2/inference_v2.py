#!/usr/bin/env python3
"""
BF-VAE v2  ·  Beat Enhancement Inference
==========================================
Handles ANY LENGTH audio input via overlap-add chunking:
  1. Detect: is this weak-beat music? (per-chunk regularity score)
  2. Enhance: run each chunk through the VAE
  3. Reconstruct: crossfade chunks back into a full-length audio

Usage:
    python inference_v2.py \
        --input   /path/to/your_song.mp3 \
        --checkpoint  7.BF_VAE_v2/checkpoints/best_model_v2.pth \
        --output  output_enhanced.wav \
        --plot    comparison.png
"""

import sys, os, argparse
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for p in [os.path.join(PROJECT_ROOT, '3.Model'), SCRIPT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_model    import MelSpectrogramVAE as VAE
from beat_loss_v2 import compute_regularity_score

SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
CHUNK_SEC   = 10.0
N_FRAMES    = 431
NORM_DIV    = 40.0
OVERLAP_SEC = 2.5          # crossfade region on each side
WEAK_BEAT_THRESHOLD = 0.45 # regularity score below this → classified as weak-beat


# ── audio ↔ mel helpers ───────────────────────────────────────────────────────

def audio_to_mel(y: np.ndarray) -> np.ndarray:
    S    = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_n  = np.clip(S_db / NORM_DIV, -1.0, 1.0)
    if S_n.shape[1] < N_FRAMES:
        S_n = np.pad(S_n, ((0, 0), (0, N_FRAMES - S_n.shape[1])))
    return S_n[:, :N_FRAMES]


def mel_to_audio(mel_np: np.ndarray) -> np.ndarray:
    power = librosa.db_to_power(mel_np * NORM_DIV)
    return librosa.feature.inverse.mel_to_audio(
        power, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=64)


def load_full_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


# ── overlap-add chunking ──────────────────────────────────────────────────────

def chunk_audio(y: np.ndarray,
                chunk_sec: float = CHUNK_SEC,
                overlap_sec: float = OVERLAP_SEC) -> list:
    """
    Split audio into overlapping chunks for VAE processing.
    Returns list of (chunk_audio, start_sample, end_sample).
    """
    chunk_len   = int(chunk_sec * SR)
    overlap_len = int(overlap_sec * SR)
    hop_len     = chunk_len - 2 * overlap_len   # non-overlapping middle
    total       = len(y)

    chunks = []
    start  = 0
    while start < total:
        end = min(start + chunk_len, total)
        chunk = y[start:end]

        # Fade-out then zero-pad last chunk so VAE sees a clean silence transition.
        # Use a long fade (up to 1.5 s) so Griffin-Lim sees a smooth decay and
        # does not generate ringing artifacts that bleed back into the valid region.
        if len(chunk) < chunk_len:
            orig_len  = len(chunk)
            fade_len  = min(int(1.5 * SR), orig_len // 2)   # up to 1.5 s fade
            fade      = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            chunk     = chunk.astype(np.float32).copy()
            chunk[-fade_len:] *= fade
            chunk = np.pad(chunk, (0, chunk_len - orig_len))

        chunks.append((chunk, start, end))
        if end >= total:
            break
        start += hop_len

    return chunks


def overlap_add(chunks_out: list, total_samples: int,
                overlap_sec: float = OVERLAP_SEC) -> np.ndarray:
    """
    Crossfade-merge VAE output chunks back into full-length audio.
    Uses raised-cosine envelope for smooth transitions.
    """
    overlap_len = int(overlap_sec * SR)
    result      = np.zeros(total_samples, dtype=np.float32)
    weight      = np.zeros(total_samples, dtype=np.float32)

    fade_in  = (1 - np.cos(np.linspace(0, np.pi, overlap_len))) / 2
    fade_out = fade_in[::-1]

    for audio_out, start, end in chunks_out:
        n = min(len(audio_out), end - start, total_samples - start)
        env = np.ones(n, dtype=np.float32)

        # Fade in at the start of the chunk
        if start > 0:
            fi = min(overlap_len, n)
            env[:fi] *= fade_in[:fi]

        # Fade out at the end of the chunk
        if end < total_samples:
            fo = min(overlap_len, n)
            env[-fo:] *= fade_out[-fo:]

        result[start:start + n] += audio_out[:n] * env
        weight[start:start + n] += env

    # Normalize by accumulated weights
    weight = np.maximum(weight, 1e-6)
    return result / weight


# ── detection ─────────────────────────────────────────────────────────────────

def detect_beat_strength(y: np.ndarray, chunk_sec: float = CHUNK_SEC) -> dict:
    """
    Analyse the full audio and return per-chunk and overall beat regularity.
    Does NOT run the VAE — just measures the input.
    """
    chunks = chunk_audio(y, chunk_sec, overlap_sec=0.0)
    scores = []
    for chunk, _, _ in chunks:
        mel = audio_to_mel(chunk)
        t   = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
        scores.append(compute_regularity_score(t))

    avg = float(np.mean(scores))
    return {
        'per_chunk_scores': scores,
        'mean_regularity':  avg,
        'is_weak_beat':     avg < WEAK_BEAT_THRESHOLD,
        'verdict': ('WEAK BEAT ✓ — enhancement recommended'
                    if avg < WEAK_BEAT_THRESHOLD
                    else 'STRONG BEAT — model may change rhythm'),
    }


# ── core enhancement ──────────────────────────────────────────────────────────

def enhance_beats(
    input_path:      str,
    checkpoint_path: str,
    output_path:     str   = None,
    plot_path:       str   = None,
    latent_dim:      int   = 128,
    device_str:      str   = 'auto',
    blend:           float = 0.5,   # 0=keep original, 1=full VAE output
) -> dict:
    """
    Full pipeline: load → detect → enhance (chunk-by-chunk) → reconstruct.
    Returns dict with all metrics and numpy arrays.
    """
    # ── device ────────────────────────────────────────────────────────────
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    print(f'Device: {device}')

    # ── load model ────────────────────────────────────────────────────────
    model = VAE(latent_dim=latent_dim).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Model: {checkpoint_path}')

    # ── load audio ────────────────────────────────────────────────────────
    print(f'\nInput : {os.path.basename(input_path)}')
    y_in  = load_full_audio(input_path)
    dur   = len(y_in) / SR
    print(f'Duration: {dur:.1f}s  ({int(dur//60)}m{int(dur%60):02d}s)')

    # ── detect beat strength BEFORE enhancement ────────────────────────────
    print('\n[1/3] Analysing beat regularity...')
    detection = detect_beat_strength(y_in)
    print(f'  Mean regularity : {detection["mean_regularity"]:.3f}')
    print(f'  Verdict         : {detection["verdict"]}')

    # ── chunk → VAE → overlap-add ──────────────────────────────────────────
    print(f'\n[2/3] Enhancing beats ({len(chunk_audio(y_in))} chunks)...')
    chunks_in  = chunk_audio(y_in)
    chunks_out = []
    reg_out_scores = []

    chunk_len = int(CHUNK_SEC * SR)
    with torch.no_grad():
        for i, (chunk, start, end) in enumerate(chunks_in):
            real_samples = end - start        # actual audio before any padding

            mel_in = audio_to_mel(chunk)                           # (128, 431)
            t_in   = torch.FloatTensor(mel_in).unsqueeze(0).unsqueeze(0).to(device)

            t_out, _, _ = model(t_in)
            if t_out.shape[-1] != N_FRAMES:
                t_out = t_out[:, :, :, :N_FRAMES]

            # Blend VAE output with original mel to preserve music identity
            mel_out_vae = t_out.squeeze().cpu().numpy()
            mel_out     = blend * mel_out_vae + (1.0 - blend) * mel_in
            audio_out   = mel_to_audio(mel_out)   # full 10 s Griffin-Lim output
            reg_out_scores.append(compute_regularity_score(t_out.cpu()))

            # If this chunk was zero-padded (tail chunk), silence everything
            # beyond the real audio length so Griffin-Lim ringing is inaudible.
            if real_samples < chunk_len:
                keep = real_samples
                fade_len = min(int(1.5 * SR), keep)
                audio_out = audio_out.copy()
                audio_out[keep:] = 0.0
                audio_out[keep - fade_len:keep] *= np.linspace(1.0, 0.0, fade_len,
                                                                dtype=np.float32)

            chunks_out.append((audio_out, start, end))
            print(f'  Chunk {i+1:2d}/{len(chunks_in)}  '
                  f'reg_out: {reg_out_scores[-1]:.3f}'
                  + (f'  [tail: {real_samples/SR:.1f}s real]'
                     if real_samples < chunk_len else ''))

    # ── reconstruct full audio ─────────────────────────────────────────────
    print('\n[3/3] Reconstructing full audio...')
    y_out = overlap_add(chunks_out, len(y_in))

    # Fade-out the final 3 s of output audio.
    # Griffin-Lim STFT edge effects degrade the last ~1 s; the 3 s window
    # ensures the noisy tail is fully silenced before the listener hears it.
    fade_samples = min(int(3.0 * SR), len(y_out) // 5)
    y_out[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    # Normalise to peak 0.85 so the output is always audible
    peak = np.max(np.abs(y_out))
    if peak > 1e-6:
        y_out = y_out / peak * 0.85

    # ── metrics ───────────────────────────────────────────────────────────
    reg_in  = detection['mean_regularity']
    reg_out = float(np.mean(reg_out_scores))

    try:
        bpm_in,  _ = librosa.beat.beat_track(y=y_in,  sr=SR, hop_length=HOP_LENGTH)
        bpm_out, _ = librosa.beat.beat_track(y=y_out, sr=SR, hop_length=HOP_LENGTH)
        bpm_in, bpm_out = float(bpm_in), float(bpm_out)
    except Exception:
        bpm_in = bpm_out = 0.0

    # MSE on mel (use first 10s for comparison)
    mel_in_ref  = audio_to_mel(y_in[:int(CHUNK_SEC * SR)])
    mel_out_ref = audio_to_mel(y_out[:int(CHUNK_SEC * SR)])
    mse = float(np.mean((mel_in_ref - mel_out_ref) ** 2))

    print(f'\n{"═"*54}')
    print(f'  INPUT  : {detection["verdict"]}')
    print(f'  Beat Regularity : {reg_in:.3f}  →  {reg_out:.3f}  '
          f'({reg_out - reg_in:+.3f})')
    print(f'  BPM             : {bpm_in:.1f}  →  {bpm_out:.1f}')
    print(f'  MSE (10s ref)   : {mse:.4f}')
    print(f'{"═"*54}')

    # ── save audio ────────────────────────────────────────────────────────
    if output_path:
        sf.write(output_path, y_out, SR)
        print(f'\nSaved enhanced audio: {output_path}')

    # ── plot ──────────────────────────────────────────────────────────────
    if plot_path:
        _plot(y_in, y_out, mel_in_ref, mel_out_ref,
              detection['per_chunk_scores'], reg_out_scores,
              reg_in, reg_out, bpm_in, bpm_out,
              detection['is_weak_beat'], plot_path,
              title=os.path.basename(input_path))
        print(f'Saved plot: {plot_path}')

    return {
        'audio_in':   y_in,
        'audio_out':  y_out,
        'reg_in':     reg_in,
        'reg_out':    reg_out,
        'reg_delta':  reg_out - reg_in,
        'bpm_in':     bpm_in,
        'bpm_out':    bpm_out,
        'mse':        mse,
        'is_weak_beat': detection['is_weak_beat'],
        'per_chunk_in':  detection['per_chunk_scores'],
        'per_chunk_out': reg_out_scores,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot(y_in, y_out, mel_in, mel_out,
          scores_in, scores_out,
          reg_in, reg_out, bpm_in, bpm_out,
          is_weak, save_path, title=''):

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    t_audio = np.linspace(0, len(y_in) / SR, len(y_in))

    # Row 0 – waveforms
    for col, (y, label, color) in enumerate([
        (y_in,  'Original waveform',  'steelblue'),
        (y_out, 'Enhanced waveform',  'tomato'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_audio, y, color=color, linewidth=0.4, alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, t_audio[-1])

    # Row 1 – Mel spectrograms (first 10s)
    for col, (mel, label) in enumerate([
        (mel_in,  f'Mel (original) — reg={reg_in:.3f}'),
        (mel_out, f'Mel (enhanced) — reg={reg_out:.3f}'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        librosa.display.specshow(mel * NORM_DIV, sr=SR, hop_length=HOP_LENGTH,
                                  x_axis='time', y_axis='mel',
                                  ax=ax, cmap='magma')
        ax.set_title(label, fontsize=10)

    # Row 2 – per-chunk beat regularity comparison
    ax = fig.add_subplot(gs[2, :])
    x = range(1, len(scores_in) + 1)
    ax.bar([i - 0.2 for i in x], scores_in,  width=0.35,
           color='steelblue', alpha=0.8, label='Original')
    ax.bar([i + 0.2 for i in x], scores_out[:len(scores_in)], width=0.35,
           color='tomato', alpha=0.8, label='Enhanced')
    ax.axhline(WEAK_BEAT_THRESHOLD, color='gray', linestyle='--',
               linewidth=1, label=f'Weak-beat threshold ({WEAK_BEAT_THRESHOLD})')
    ax.set_xlabel('Chunk index')
    ax.set_ylabel('Beat Regularity Score')
    ax.set_title('Per-chunk beat regularity: Original vs Enhanced', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    verdict = 'WEAK BEAT → Enhancement applied' if is_weak else 'STRONG BEAT detected'
    fig.suptitle(
        f'{title}  |  {verdict}\n'
        f'Regularity: {reg_in:.3f} → {reg_out:.3f} ({reg_out-reg_in:+.3f})  '
        f'BPM: {bpm_in:.1f} → {bpm_out:.1f}',
        fontsize=12, y=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BF-VAE v2 — Beat Enhancement')
    p.add_argument('--input',      required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output',     default=None, help='Output .wav path')
    p.add_argument('--plot',       default=None, help='Output .png path')
    p.add_argument('--latent_dim', type=int,   default=128)
    p.add_argument('--device',     default='auto')
    p.add_argument('--blend',      type=float, default=0.5,
                   help='Mix ratio: 0=original only, 1=full VAE (default 0.5)')
    return p.parse_args()


if __name__ == '__main__':
    args  = parse_args()
    base  = os.path.splitext(os.path.basename(args.input))[0]
    out_w = args.output or f'{base}_enhanced.wav'
    out_p = args.plot   or f'{base}_comparison.png'

    enhance_beats(
        input_path      = args.input,
        checkpoint_path = args.checkpoint,
        output_path     = out_w,
        plot_path       = out_p,
        latent_dim      = args.latent_dim,
        device_str      = args.device,
        blend           = args.blend,
    )
