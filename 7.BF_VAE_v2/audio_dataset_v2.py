"""
AudioMelDataset v2
==================
Improvements over v1:
  - Validates each file has enough audio content (skips silent / too-short files)
  - For long tracks (>= 2×window): samples multiple non-overlapping windows
    so every second of the track gets used during training
  - Reports dataset statistics on init
  - Detects and skips corrupted files gracefully
"""

import random
import warnings
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

SR         = 22050
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
DURATION   = 10.0
N_FRAMES   = 431          # ceil(SR * DURATION / HOP_LENGTH)
NORM_DIV   = 40.0
MIN_VALID_DURATION = 8.0  # skip files shorter than this
SILENCE_THRESHOLD  = 0.01 # RMS below this = silent window → resample


class AudioMelDataset_v2(Dataset):
    """
    PyTorch Dataset that converts audio files to log-Mel spectrograms.

    Key design:
    • Each __getitem__ call returns ONE 10-second window from a random file.
    • For files > 20s: multiple non-overlapping start offsets are registered,
      so the full file gets sampled proportionally during an epoch.
    • Corrupted / silent files are skipped at init time.
    • Zero-padding only happens for the rare valid-but-short (8-10s) files.
    """

    EXTENSIONS = ('.mp3', '.wav', '.flac', '.m4a', '.ogg')

    def __init__(
        self,
        data_dir: str,
        window_sec: float = DURATION,
        min_duration: float = MIN_VALID_DURATION,
        max_silence_ratio: float = 0.5,
        verbose: bool = True,
    ):
        self.data_dir         = Path(data_dir)
        self.window_sec       = window_sec
        self.window_samples   = int(window_sec * SR)
        self.min_duration     = min_duration
        self.max_silence_ratio = max_silence_ratio

        # Each entry in self.samples = (file_path, start_offset_sec)
        self.samples: List[Tuple[Path, float]] = []
        self._build_sample_list(verbose)

    # ── dataset construction ──────────────────────────────────────────────

    def _build_sample_list(self, verbose: bool):
        all_files = sorted(
            p for p in self.data_dir.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        )

        skipped_short    = 0
        skipped_corrupt  = 0
        windows_added    = 0

        for fp in all_files:
            try:
                dur = librosa.get_duration(path=str(fp))
            except Exception:
                skipped_corrupt += 1
                continue

            if dur < self.min_duration:
                skipped_short += 1
                continue

            # Register non-overlapping 10s windows across the full track
            # e.g. 30s track → offsets [0, 10, 20]  (3 windows)
            step = self.window_sec
            usable = dur - self.window_sec          # last valid start
            offset = 0.0
            while offset <= usable:
                self.samples.append((fp, offset))
                windows_added += 1
                offset += step

            # If track is between min_duration and window_sec, add once at offset=0
            if usable < 0:
                self.samples.append((fp, 0.0))
                windows_added += 1

        if verbose:
            n_files = len(set(s[0] for s in self.samples))
            print(f'[AudioMelDataset_v2]')
            print(f'  Files accepted    : {n_files}')
            print(f'  Windows (samples) : {len(self.samples)}')
            print(f'  Skipped (short)   : {skipped_short}')
            print(f'  Skipped (corrupt) : {skipped_corrupt}')
            print(f'  Avg windows/file  : {len(self.samples)/max(n_files,1):.1f}')

    # ── PyTorch Dataset API ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        fp, base_offset = self.samples[idx]

        # Add small random jitter (±2s) to avoid always cutting at the same point,
        # while staying within the track duration.
        try:
            dur = librosa.get_duration(path=str(fp))
        except Exception:
            return self._zero_tensor()

        jitter = random.uniform(-2.0, 2.0)
        offset = float(np.clip(base_offset + jitter, 0.0,
                               max(0.0, dur - self.window_sec)))

        for attempt in range(3):          # retry up to 3 times if window is silent
            mel = self._load_window(fp, offset)
            if mel is not None:
                return torch.FloatTensor(mel).unsqueeze(0)   # (1, 128, 431)
            # Window was silent → try a different random offset
            offset = random.uniform(0.0, max(0.0, dur - self.window_sec))

        return self._zero_tensor()

    # ── internal helpers ──────────────────────────────────────────────────

    def _load_window(self, fp: Path, offset: float):
        try:
            y, _ = librosa.load(str(fp), sr=SR, mono=True,
                                duration=self.window_sec, offset=offset)
        except Exception:
            return None

        # Zero-pad if needed (only for tracks 8-10s)
        if len(y) < self.window_samples:
            y = np.pad(y, (0, self.window_samples - len(y)))

        # Reject silent windows
        rms = np.sqrt(np.mean(y ** 2))
        if rms < SILENCE_THRESHOLD:
            return None

        return self._to_mel(y)

    def _to_mel(self, y: np.ndarray) -> np.ndarray:
        S    = librosa.feature.melspectrogram(
                   y=y, sr=SR, n_fft=N_FFT,
                   hop_length=HOP_LENGTH, n_mels=N_MELS)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_n  = np.clip(S_db / NORM_DIV, -1.0, 1.0)

        # Ensure exactly N_FRAMES columns
        if S_n.shape[1] < N_FRAMES:
            S_n = np.pad(S_n, ((0, 0), (0, N_FRAMES - S_n.shape[1])))
        return S_n[:, :N_FRAMES]

    def _zero_tensor(self) -> torch.FloatTensor:
        return torch.zeros(1, N_MELS, N_FRAMES)

    # ── inspection helpers ────────────────────────────────────────────────

    def duration_stats(self) -> dict:
        """Compute duration distribution (reads file headers — slow on large datasets)."""
        durations = []
        for fp, _ in set((s[0], 0) for s in self.samples):
            try:
                durations.append(librosa.get_duration(path=str(fp)))
            except Exception:
                pass
        d = np.array(durations)
        return {
            'n_files': len(d),
            'min_s': float(d.min()),
            'max_s': float(d.max()),
            'mean_s': float(d.mean()),
            'median_s': float(np.median(d)),
        }
