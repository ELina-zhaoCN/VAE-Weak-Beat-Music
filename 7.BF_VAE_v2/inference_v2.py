#!/usr/bin/env python3
"""
BF-VAE v2  ·  Inference / Beat Enhancement Test
=================================================
Input : weak-beat audio (MP3 / WAV / FLAC)
Output: audio with stronger beat pattern

Usage:
    python inference_v2.py \
        --input  path/to/ambient.mp3 \
        --checkpoint  checkpoints_v2/best_model_v2.pth \
        --output  output_with_beats.wav \
        --plot    comparison.png

What it shows:
    • Mel-spectrogram  (original vs reconstructed)
    • Low-freq energy envelope + autocorrelation
    • Beat regularity score and BPM before/after
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

# ── project path setup ────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for p in [os.path.join(PROJECT_ROOT, '3.Model'), SCRIPT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_model    import MelSpectrogramVAE as VAE
from beat_loss_v2 import compute_regularity_score

# ── audio constants ───────────────────────────────────────────────────────────
SR         = 22050
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
DURATION   = 10
N_FRAMES   = 431
NORM_DIV   = 40.0


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


def load_audio_clip(path: str, offset: float = 0.0) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, duration=DURATION,
                        offset=offset, mono=True)
    target = SR * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]


# ── beat metrics ──────────────────────────────────────────────────────────────
def get_bpm(y: np.ndarray) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    return float(tempo)


def get_autocorr(mel_np: np.ndarray,
                 n_low: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Returns (acf, energy_envelope)."""
    energy = np.mean(np.abs(mel_np[:n_low, :]), axis=0)
    en = (energy - energy.mean()) / (energy.std() + 1e-8)
    n  = len(en)
    f  = np.fft.fft(en, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf = acf / (acf[0] + 1e-8)
    return acf, energy


# ── main inference function ───────────────────────────────────────────────────
def enhance_beats(input_path: str,
                  checkpoint_path: str,
                  output_path: str = None,
                  plot_path: str = None,
                  offset: float = 0.0,
                  latent_dim: int = 128,
                  device_str: str = 'auto') -> dict:
    """
    Load weak-beat audio → run through BF-VAE v2 → return beat-enhanced audio.

    Returns a dict with all metrics and numpy audio arrays.
    """
    # ── device ────────────────────────────────────────────────────────────
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f'Device: {device}')

    # ── load model ────────────────────────────────────────────────────────
    model = VAE(latent_dim=latent_dim).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Model loaded from {checkpoint_path}')

    # ── load & process input ──────────────────────────────────────────────
    print(f'\nProcessing: {os.path.basename(input_path)}')
    audio_in = load_audio_clip(input_path, offset=offset)
    mel_in   = audio_to_mel(audio_in)

    mel_tensor = torch.FloatTensor(mel_in).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        mel_out_tensor, mu, logvar = model(mel_tensor)
        if mel_out_tensor.shape[-1] != N_FRAMES:
            mel_out_tensor = mel_out_tensor[:, :, :, :N_FRAMES]

    mel_out = mel_out_tensor.squeeze().cpu().numpy()

    # ── reconstruct audio ─────────────────────────────────────────────────
    audio_out = mel_to_audio(mel_out)

    # ── metrics ───────────────────────────────────────────────────────────
    reg_in  = compute_regularity_score(mel_tensor.cpu())
    reg_out = compute_regularity_score(mel_out_tensor.cpu())
    bpm_in  = get_bpm(audio_in)
    bpm_out = get_bpm(audio_out)
    mse     = float(np.mean((mel_in - mel_out) ** 2))

    acf_in,  env_in  = get_autocorr(mel_in)
    acf_out, env_out = get_autocorr(mel_out)

    print(f'\n{"="*52}')
    print(f'  Beat Regularity : {reg_in:.3f}  →  {reg_out:.3f}  '
          f'({"+" if reg_out>reg_in else ""}{reg_out-reg_in:+.3f})')
    print(f'  BPM             : {bpm_in:.1f}  →  {bpm_out:.1f}')
    print(f'  MSE             : {mse:.4f}')
    print(f'{"="*52}')

    # ── save audio ────────────────────────────────────────────────────────
    if output_path:
        sf.write(output_path, audio_out, SR)
        print(f'\nOutput saved: {output_path}')

    # ── plot ──────────────────────────────────────────────────────────────
    if plot_path:
        _plot_comparison(mel_in, mel_out, env_in, env_out, acf_in, acf_out,
                         reg_in, reg_out, bpm_in, bpm_out, plot_path,
                         title=os.path.basename(input_path))
        print(f'Plot saved: {plot_path}')

    return {
        'audio_in':   audio_in,
        'audio_out':  audio_out,
        'mel_in':     mel_in,
        'mel_out':    mel_out,
        'reg_in':     reg_in,
        'reg_out':    reg_out,
        'reg_delta':  reg_out - reg_in,
        'bpm_in':     bpm_in,
        'bpm_out':    bpm_out,
        'mse':        mse,
    }


def _plot_comparison(mel_in, mel_out, env_in, env_out, acf_in, acf_out,
                     reg_in, reg_out, bpm_in, bpm_out, save_path, title=''):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    frames = np.arange(N_FRAMES)

    # ── Row 0: Mel spectrograms ────────────────────────────────────────────
    for col, (mel, label) in enumerate([(mel_in,  'Original (Weak Beat)'),
                                         (mel_out, 'Enhanced (Strong Beat)')]):
        ax = fig.add_subplot(gs[0, col])
        librosa.display.specshow(mel * NORM_DIV, sr=SR, hop_length=HOP_LENGTH,
                                  x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Mel)')

    # ── Row 1: Low-freq energy envelope ───────────────────────────────────
    t = frames * HOP_LENGTH / SR
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, env_in,  alpha=0.8, label=f'Original  (regularity={reg_in:.3f})',  color='steelblue')
    ax.plot(t, env_out, alpha=0.8, label=f'Enhanced  (regularity={reg_out:.3f})', color='tomato')
    ax.set_title('Low-Frequency Energy Envelope (beat region: 0–250 Hz)', fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Autocorrelation ─────────────────────────────────────────────
    # BPM range [60,240] → lag in frames
    lag_min = max(1, int(round(SR / (HOP_LENGTH * 240 / 60))))
    lag_max = min(N_FRAMES - 1, int(round(SR / (HOP_LENGTH * 60 / 60))))

    lag_axis = np.arange(N_FRAMES) * HOP_LENGTH / SR
    for col, (acf, label, color) in enumerate([
        (acf_in,  f'Original  BPM≈{bpm_in:.0f}',  'steelblue'),
        (acf_out, f'Enhanced  BPM≈{bpm_out:.0f}', 'tomato'),
    ]):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(lag_axis, acf, color=color, linewidth=1.2)
        ax.axvspan(lag_min * HOP_LENGTH / SR, lag_max * HOP_LENGTH / SR,
                   alpha=0.12, color='gold', label='60–240 BPM search window')
        ax.set_xlim(0, lag_axis[lag_max + 10] if lag_max + 10 < len(lag_axis) else lag_axis[-1])
        ax.set_title(f'Autocorrelation – {label}', fontsize=10)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Correlation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    delta_str = f'{reg_out - reg_in:+.3f}'
    fig.suptitle(f'{title}\n'
                 f'Beat Regularity: {reg_in:.3f} → {reg_out:.3f} ({delta_str})   '
                 f'BPM: {bpm_in:.1f} → {bpm_out:.1f}',
                 fontsize=13, y=0.99)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='BF-VAE v2 Beat Enhancement Inference')
    p.add_argument('--input',      required=True,  help='Input weak-beat audio file')
    p.add_argument('--checkpoint', required=True,  help='Path to best_model_v2.pth')
    p.add_argument('--output',     default=None,   help='Save enhanced audio (.wav)')
    p.add_argument('--plot',       default=None,   help='Save comparison plot (.png)')
    p.add_argument('--offset',     type=float, default=0.0, help='Audio clip offset (s)')
    p.add_argument('--latent_dim', type=int,   default=128)
    p.add_argument('--device',     default='auto', help='cuda / cpu / auto')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Auto-generate output paths if not given
    base    = os.path.splitext(os.path.basename(args.input))[0]
    out_wav = args.output or f'{base}_beat_enhanced.wav'
    out_png = args.plot   or f'{base}_comparison.png'

    results = enhance_beats(
        input_path      = args.input,
        checkpoint_path = args.checkpoint,
        output_path     = out_wav,
        plot_path       = out_png,
        offset          = args.offset,
        latent_dim      = args.latent_dim,
        device_str      = args.device,
    )

    print(f'\nDone!')
    print(f'  Output audio : {out_wav}')
    print(f'  Plot         : {out_png}')
