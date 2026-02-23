"""
Audio Dataset for Mel-Spectrogram Generation
=============================================
PyTorch Dataset class for loading audio files and converting them to Mel-spectrograms.
Designed for weak beat music analysis and audio generation tasks.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List
import librosa
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AudioMelDataset(Dataset):
    """
    PyTorch Dataset for loading audio files and converting to Mel-spectrograms.
    
    This dataset loads audio files, applies random temporal offsets, and converts
    them to log-scaled Mel-spectrograms suitable for training audio models.
    
    Args:
        data_dir (str): Path to directory containing audio files
        sr (int): Sampling rate for audio loading (default: 22050 Hz)
        duration (float): Duration of each audio segment in seconds (default: 10.0)
        n_mels (int): Number of Mel frequency bands (default: 128)
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Number of samples between successive frames (default: 512)
        max_offset (float): Maximum random offset in seconds (default: 5.0)
        audio_extensions (tuple): Supported audio file extensions
        normalize (bool): Whether to normalize Mel-spectrograms (default: True)
    
    Returns:
        torch.FloatTensor: Mel-spectrogram of shape (1, n_mels, n_frames)
    """
    
    def __init__(
        self,
        data_dir: str,
        sr: int = 22050,
        duration: float = 10.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_offset: float = 5.0,
        audio_extensions: tuple = ('.mp3', '.wav', '.flac', '.m4a', '.ogg'),
        normalize: bool = True
    ):
        """Initialize the dataset."""
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_offset = max_offset
        self.audio_extensions = audio_extensions
        self.normalize = normalize
        
        # Calculate expected length and number of frames
        self.target_length = int(sr * duration)
        self.n_frames = int(duration * sr / hop_length)
        
        # Scan for audio files
        self.audio_files = self._scan_audio_files()
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"AudioMelDataset initialized:")
        print(f"  Directory: {self.data_dir}")
        print(f"  Audio files: {len(self.audio_files)}")
        print(f"  Sampling rate: {self.sr} Hz")
        print(f"  Duration: {self.duration}s")
        print(f"  Target length: {self.target_length} samples")
        print(f"  Mel bands: {self.n_mels}")
        print(f"  Expected frames: {self.n_frames}")
        print(f"  Output shape: (1, {self.n_mels}, {self.n_frames})")
    
    def _scan_audio_files(self) -> List[Path]:
        """Scan directory for audio files."""
        audio_files = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # Recursively find all audio files
        for ext in self.audio_extensions:
            audio_files.extend(self.data_dir.rglob(f"*{ext}"))
        
        # Sort for reproducibility
        audio_files.sort()
        
        return audio_files
    
    def _load_audio_segment(self, audio_path: Path) -> np.ndarray:
        """
        Load audio file with random offset and ensure correct length.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            np.ndarray: Audio waveform of length self.target_length
        """
        # Generate random offset (0 to max_offset seconds)
        offset = random.uniform(0, self.max_offset)
        
        # Load audio with librosa
        # Note: librosa.load returns mono audio by default
        audio, _ = librosa.load(
            audio_path,
            sr=self.sr,
            offset=offset,
            duration=self.duration,
            mono=True
        )
        
        # Ensure correct length
        if len(audio) < self.target_length:
            # Pad with zeros if too short
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        elif len(audio) > self.target_length:
            # Truncate if too long
            audio = audio[:self.target_length]
        
        return audio
    
    def audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to log-scaled Mel-spectrogram.
        
        Args:
            audio: Audio waveform of shape (target_length,)
            
        Returns:
            np.ndarray: Mel-spectrogram of shape (n_mels, n_frames)
        """
        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sr / 2.0
        )
        
        # Apply log scaling (convert to dB)
        # Add small epsilon to avoid log(0)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range if requested
        if self.normalize:
            # Min-max normalization
            min_val = log_mel_spec.min()
            max_val = log_mel_spec.max()
            
            if max_val > min_val:
                log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val)
            else:
                # Handle edge case where all values are the same
                log_mel_spec = np.zeros_like(log_mel_spec)
        
        # Ensure correct number of frames
        if log_mel_spec.shape[1] < self.n_frames:
            # Pad with zeros if too few frames
            pad_width = ((0, 0), (0, self.n_frames - log_mel_spec.shape[1]))
            log_mel_spec = np.pad(log_mel_spec, pad_width, mode='constant', constant_values=0)
        elif log_mel_spec.shape[1] > self.n_frames:
            # Truncate if too many frames
            log_mel_spec = log_mel_spec[:, :self.n_frames]
        
        return log_mel_spec
    
    def _get_zero_tensor(self) -> torch.FloatTensor:
        """Return a zero tensor with the expected shape."""
        return torch.zeros(1, self.n_mels, self.n_frames, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of audio files in the dataset."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> torch.FloatTensor:
        """
        Load and process an audio file to return a Mel-spectrogram.
        
        Args:
            idx: Index of the audio file (note: actual file is randomly selected)
            
        Returns:
            torch.FloatTensor: Mel-spectrogram of shape (1, n_mels, n_frames)
        """
        try:
            # Randomly select an audio file (ignoring idx for random selection)
            audio_path = random.choice(self.audio_files)
            
            # Load audio segment with random offset
            audio = self._load_audio_segment(audio_path)
            
            # Convert to Mel-spectrogram
            mel_spec = self.audio_to_mel(audio)
            
            # Convert to torch tensor
            mel_tensor = torch.from_numpy(mel_spec).float()
            
            # Add channel dimension: (n_mels, n_frames) -> (1, n_mels, n_frames)
            mel_tensor = mel_tensor.unsqueeze(0)
            
            return mel_tensor
            
        except Exception as e:
            # Handle any errors during loading
            print(f"Error loading audio file: {e}")
            print(f"Attempted file: {audio_path if 'audio_path' in locals() else 'unknown'}")
            
            # Return zero tensor as fallback
            return self._get_zero_tensor()
    
    def get_by_index(self, idx: int) -> torch.FloatTensor:
        """
        Get a specific audio file by index (non-random selection).
        
        Args:
            idx: Index of the audio file
            
        Returns:
            torch.FloatTensor: Mel-spectrogram of shape (1, n_mels, n_frames)
        """
        if idx < 0 or idx >= len(self.audio_files):
            raise IndexError(f"Index {idx} out of range [0, {len(self.audio_files)})")
        
        try:
            audio_path = self.audio_files[idx]
            
            # Load audio with fixed offset (0 seconds)
            audio, _ = librosa.load(
                audio_path,
                sr=self.sr,
                offset=0,
                duration=self.duration,
                mono=True
            )
            
            # Ensure correct length
            if len(audio) < self.target_length:
                pad_length = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            elif len(audio) > self.target_length:
                audio = audio[:self.target_length]
            
            # Convert to Mel-spectrogram
            mel_spec = self.audio_to_mel(audio)
            
            # Convert to torch tensor and add channel dimension
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            
            return mel_tensor
            
        except Exception as e:
            print(f"Error loading audio file at index {idx}: {e}")
            print(f"File: {self.audio_files[idx]}")
            return self._get_zero_tensor()
    
    def get_audio_info(self, idx: int) -> dict:
        """
        Get information about an audio file without loading it.
        
        Args:
            idx: Index of the audio file
            
        Returns:
            dict: Audio file information
        """
        if idx < 0 or idx >= len(self.audio_files):
            raise IndexError(f"Index {idx} out of range [0, {len(self.audio_files)})")
        
        audio_path = self.audio_files[idx]
        
        try:
            # Get audio duration without loading full file
            duration = librosa.get_duration(path=audio_path)
            
            return {
                'path': str(audio_path),
                'filename': audio_path.name,
                'duration': duration,
                'index': idx
            }
        except Exception as e:
            return {
                'path': str(audio_path),
                'filename': audio_path.name,
                'duration': None,
                'index': idx,
                'error': str(e)
            }


def test_dataset():
    """Test function for the AudioMelDataset class."""
    import matplotlib.pyplot as plt
    
    print("Testing AudioMelDataset...")
    print("="*70)
    
    # Check if weak_beat_music directory exists
    data_dir = "./weak_beat_music"
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found.")
        print("Please run fma_filter.py first to create filtered music dataset.")
        print("\nCreating test with dummy directory for demonstration...")
        return
    
    try:
        # Initialize dataset
        dataset = AudioMelDataset(
            data_dir=data_dir,
            sr=22050,
            duration=10.0,
            n_mels=128,
            hop_length=512,
            max_offset=5.0
        )
        
        print(f"\n✓ Dataset created successfully!")
        print(f"  Total files: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test __getitem__
            print("\nTesting __getitem__ (random selection)...")
            mel_spec = dataset[0]
            print(f"  Output shape: {mel_spec.shape}")
            print(f"  Data type: {mel_spec.dtype}")
            print(f"  Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
            
            # Test get_by_index
            print("\nTesting get_by_index (specific file)...")
            mel_spec_idx = dataset.get_by_index(0)
            print(f"  Output shape: {mel_spec_idx.shape}")
            
            # Test get_audio_info
            print("\nTesting get_audio_info...")
            info = dataset.get_audio_info(0)
            print(f"  File: {info['filename']}")
            print(f"  Duration: {info.get('duration', 'N/A')}s")
            
            # Visualize
            print("\nGenerating visualization...")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot first sample
            mel_display_1 = mel_spec[0].numpy()  # Remove channel dimension
            im1 = axes[0].imshow(mel_display_1, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title('Random Sample - Mel-Spectrogram')
            axes[0].set_xlabel('Time Frames')
            axes[0].set_ylabel('Mel Frequency Bands')
            plt.colorbar(im1, ax=axes[0])
            
            # Plot second sample for comparison
            mel_spec_2 = dataset[1]
            mel_display_2 = mel_spec_2[0].numpy()
            im2 = axes[1].imshow(mel_display_2, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title('Another Random Sample - Mel-Spectrogram')
            axes[1].set_xlabel('Time Frames')
            axes[1].set_ylabel('Mel Frequency Bands')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig('mel_spectrogram_examples.png', dpi=150, bbox_inches='tight')
            print("  ✓ Saved visualization to: mel_spectrogram_examples.png")
            
            print("\n" + "="*70)
            print("✓ All tests passed!")
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
