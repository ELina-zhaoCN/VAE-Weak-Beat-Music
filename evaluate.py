#!/usr/bin/env python3
"""
Evaluation Script for Weak Beat Music VAE
==========================================
Loads trained model, generates reconstructions, and creates visualizations
comparing original and reconstructed audio.

Usage:
    python evaluate.py --checkpoint ./checkpoints/best_model.pt --data_dir ./weak_beat_music
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import custom modules
try:
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import VAE model
    try:
        from vae_model import MelSpectrogramVAE
    except ImportError:
        from model import MelSpectrogramVAE
    
    # Import dataset
    try:
        from music_dataset.audio_dataset import AudioMelDataset
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent / 'music_dataset'))
            from audio_dataset import AudioMelDataset
        except ImportError:
            print("Error: Could not import AudioMelDataset")
            sys.exit(1)
    
    # Import beat loss for autocorrelation
    try:
        from beat_loss import compute_autocorrelation_fft, beat_loss
    except ImportError:
        print("Warning: beat_loss not found. Autocorrelation plots will be limited.")
        compute_autocorrelation_fft = None
        beat_loss = None

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained VAE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./weak_beat_music',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to evaluate')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension (must match trained model)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save reconstructed audio as WAV files')
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def load_model(checkpoint_path: str, latent_dim: int, device: torch.device) -> MelSpectrogramVAE:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        latent_dim: Latent space dimension
        device: Device to load model on
    
    Returns:
        model: Loaded VAE model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = MelSpectrogramVAE(latent_dim=latent_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    print(f"Checkpoint epoch: {epoch}")
    if metrics:
        print(f"Validation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return model


def mel_to_audio(mel_spec: torch.Tensor, 
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_iter: int = 32) -> np.ndarray:
    """
    Convert Mel-spectrogram to audio using Griffin-Lim algorithm.
    
    Args:
        mel_spec: Mel-spectrogram tensor (1, n_mels, n_frames) or (n_mels, n_frames)
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations
    
    Returns:
        audio: Audio waveform
    """
    # Convert to numpy
    if isinstance(mel_spec, torch.Tensor):
        mel_spec_np = mel_spec.cpu().numpy()
        if mel_spec_np.ndim == 3:
            mel_spec_np = mel_spec_np[0]  # Remove channel dimension
    else:
        mel_spec_np = mel_spec
    
    # De-normalize from [0, 1] to dB scale
    # Assuming the mel-spectrogram was normalized during dataset creation
    mel_spec_db = mel_spec_np * 80.0 - 80.0  # Approximate denormalization
    
    # Convert dB to power
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    
    # Convert mel to linear spectrogram
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec_power,
        sr=sr,
        n_fft=n_fft
    )
    
    # Griffin-Lim to reconstruct audio
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft
    )
    
    return audio


def estimate_bpm(mel_spec: torch.Tensor, sr: int = 22050, hop_length: int = 512) -> float:
    """
    Estimate BPM from mel-spectrogram using librosa.
    
    Args:
        mel_spec: Mel-spectrogram (1, n_mels, n_frames) or (n_mels, n_frames)
        sr: Sampling rate
        hop_length: Hop length
    
    Returns:
        bpm: Estimated BPM
    """
    # Convert to audio first
    audio = mel_to_audio(mel_spec, sr=sr, hop_length=hop_length, n_iter=16)
    
    # Use librosa beat tracking
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
        # tempo is returned as a scalar in recent librosa versions, or array in older ones
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if len(tempo) > 0 else 0.0
        return float(tempo)
    except Exception as e:
        print(f"  Warning: BPM estimation failed: {e}")
        return 0.0


def compute_autocorrelation(mel_spec: torch.Tensor) -> np.ndarray:
    """
    Compute autocorrelation of energy envelope for beat analysis.
    
    Args:
        mel_spec: Mel-spectrogram (1, n_mels, n_frames) or (n_mels, n_frames)
    
    Returns:
        autocorr: Autocorrelation values
    """
    if compute_autocorrelation_fft is not None:
        # Use the function from beat_loss.py
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dim
        
        # Extract low-frequency energy envelope
        low_freq = mel_spec[:, :, :10, :]  # First 10 mel bands
        energy = torch.mean(low_freq, dim=2).squeeze(1)  # (1, n_frames)
        
        # Compute autocorrelation
        autocorr = compute_autocorrelation_fft(energy)
        return autocorr[0].cpu().numpy()
    else:
        # Fallback: simple numpy autocorrelation
        if mel_spec.dim() == 3:
            mel_spec = mel_spec[0]
        
        mel_np = mel_spec.cpu().numpy()
        energy = np.mean(mel_np[:10], axis=0)
        
        # Normalize
        energy = energy - np.mean(energy)
        
        # Compute autocorrelation
        autocorr = np.correlate(energy, energy, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        
        return autocorr


def plot_comparison(original_mel: torch.Tensor,
                   reconstructed_mel: torch.Tensor,
                   original_bpm: float,
                   reconstructed_bpm: float,
                   sample_idx: int,
                   output_path: Path,
                   sr: int = 22050,
                   hop_length: int = 512):
    """
    Create comprehensive visualization comparing original and reconstructed audio.
    
    Args:
        original_mel: Original mel-spectrogram (1, n_mels, n_frames)
        reconstructed_mel: Reconstructed mel-spectrogram (1, n_mels, n_frames)
        original_bpm: Original BPM
        reconstructed_bpm: Reconstructed BPM
        sample_idx: Sample index for labeling
        output_path: Path to save figure
        sr: Sampling rate
        hop_length: Hop length
    """
    # Convert to numpy for plotting
    orig_np = original_mel[0].cpu().numpy() if original_mel.dim() == 3 else original_mel.cpu().numpy()
    recon_np = reconstructed_mel[0].cpu().numpy() if reconstructed_mel.dim() == 3 else reconstructed_mel.cpu().numpy()
    
    # Compute autocorrelations
    orig_autocorr = compute_autocorrelation(original_mel)
    recon_autocorr = compute_autocorrelation(reconstructed_mel)
    
    # Convert to audio for waveform plots
    print(f"  Converting to audio (Griffin-Lim)...")
    orig_audio = mel_to_audio(original_mel, sr=sr, hop_length=hop_length)
    recon_audio = mel_to_audio(reconstructed_mel, sr=sr, hop_length=hop_length)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Waveforms
    ax1 = fig.add_subplot(gs[0, 0])
    time_orig = np.arange(len(orig_audio)) / sr
    ax1.plot(time_orig, orig_audio, linewidth=0.5, color='blue', alpha=0.7)
    ax1.set_title(f'Original Waveform (BPM: {original_bpm:.1f})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_orig[-1])
    
    ax2 = fig.add_subplot(gs[0, 1])
    time_recon = np.arange(len(recon_audio)) / sr
    ax2.plot(time_recon, recon_audio, linewidth=0.5, color='red', alpha=0.7)
    ax2.set_title(f'Reconstructed Waveform (BPM: {reconstructed_bpm:.1f})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_recon[-1])
    
    # Row 2: Mel-spectrograms
    ax3 = fig.add_subplot(gs[1, 0])
    img1 = librosa.display.specshow(orig_np, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='mel', cmap='viridis', ax=ax3)
    ax3.set_title('Original Mel-Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=ax3, format='%+2.0f dB')
    
    ax4 = fig.add_subplot(gs[1, 1])
    img2 = librosa.display.specshow(recon_np, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='mel', cmap='viridis', ax=ax4)
    ax4.set_title('Reconstructed Mel-Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=ax4, format='%+2.0f dB')
    
    # Row 3: Autocorrelations (Beat analysis)
    ax5 = fig.add_subplot(gs[2, 0])
    lag_time_orig = np.arange(len(orig_autocorr)) * hop_length / sr
    ax5.plot(lag_time_orig, orig_autocorr, linewidth=2, color='blue')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_title('Original Autocorrelation (Beat Regularity)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Lag (s)')
    ax5.set_ylabel('Correlation')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 5)  # Show first 5 seconds
    max_corr_orig = np.max(orig_autocorr[10:]) if len(orig_autocorr) > 10 else 0
    ax5.text(0.02, 0.98, f'Max correlation: {max_corr_orig:.3f}',
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax6 = fig.add_subplot(gs[2, 1])
    lag_time_recon = np.arange(len(recon_autocorr)) * hop_length / sr
    ax6.plot(lag_time_recon, recon_autocorr, linewidth=2, color='red')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.set_title('Reconstructed Autocorrelation (Beat Regularity)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Lag (s)')
    ax6.set_ylabel('Correlation')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 5)
    max_corr_recon = np.max(recon_autocorr[10:]) if len(recon_autocorr) > 10 else 0
    ax6.text(0.02, 0.98, f'Max correlation: {max_corr_recon:.3f}',
             transform=ax6.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Sample {sample_idx} - Original vs Reconstructed Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_path}")


def evaluate_model(model: MelSpectrogramVAE,
                  dataset: AudioMelDataset,
                  num_samples: int,
                  output_dir: Path,
                  device: torch.device,
                  save_audio: bool = False,
                  sr: int = 22050,
                  hop_length: int = 512):
    """
    Evaluate model on test samples and create visualizations.
    
    Args:
        model: Trained VAE model
        dataset: Test dataset
        num_samples: Number of samples to evaluate
        output_dir: Output directory for results
        device: Device to use
        save_audio: Whether to save reconstructed audio
        sr: Sampling rate
        hop_length: Hop length
    """
    print(f"\nEvaluating {num_samples} samples...")
    print("="*70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    vis_dir = output_dir / 'visualizations'
    audio_dir = output_dir / 'audio'
    vis_dir.mkdir(exist_ok=True)
    if save_audio:
        audio_dir.mkdir(exist_ok=True)
    
    # Evaluation metrics
    mse_scores = []
    bpm_diffs = []
    beat_loss_scores = []
    
    results_summary = []
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            print(f"\nSample {i+1}/{num_samples}")
            print("-"*70)
            
            # Get sample
            original_mel = dataset[i].unsqueeze(0).to(device)  # Add batch dimension
            
            # Reconstruct
            print(f"  Reconstructing...")
            reconstructed_mel, mu, logvar = model(original_mel)
            
            # Compute MSE
            mse = F.mse_loss(reconstructed_mel, original_mel).item()
            mse_scores.append(mse)
            print(f"  MSE: {mse:.4f}")
            
            # Estimate BPM
            print(f"  Estimating BPM (this may take a moment)...")
            orig_bpm = estimate_bpm(original_mel[0], sr=sr, hop_length=hop_length)
            recon_bpm = estimate_bpm(reconstructed_mel[0], sr=sr, hop_length=hop_length)
            bpm_diff = abs(orig_bpm - recon_bpm)
            bpm_diffs.append(bpm_diff)
            
            print(f"  Original BPM: {orig_bpm:.1f}")
            print(f"  Reconstructed BPM: {recon_bpm:.1f}")
            print(f"  BPM difference: {bpm_diff:.1f}")
            
            # Compute beat loss (if available)
            if beat_loss is not None:
                beat_score_orig = beat_loss(original_mel, normalize=True).item()
                beat_score_recon = beat_loss(reconstructed_mel, normalize=True).item()
                beat_loss_scores.append((beat_score_orig, beat_score_recon))
                print(f"  Beat regularity (original): {beat_score_orig:.4f}")
                print(f"  Beat regularity (reconstructed): {beat_score_recon:.4f}")
            else:
                beat_score_orig = 0.0
                beat_score_recon = 0.0
            
            # Create visualization
            print(f"  Creating visualization...")
            vis_path = vis_dir / f'sample_{i+1}_comparison.png'
            plot_comparison(
                original_mel[0], reconstructed_mel[0],
                orig_bpm, recon_bpm,
                i+1, vis_path, sr, hop_length
            )
            
            # Save audio (if requested)
            if save_audio:
                print(f"  Saving audio files...")
                orig_audio = mel_to_audio(original_mel[0], sr=sr, hop_length=hop_length)
                recon_audio = mel_to_audio(reconstructed_mel[0], sr=sr, hop_length=hop_length)
                
                import soundfile as sf
                sf.write(audio_dir / f'sample_{i+1}_original.wav', orig_audio, sr)
                sf.write(audio_dir / f'sample_{i+1}_reconstructed.wav', recon_audio, sr)
                print(f"    Saved: sample_{i+1}_original.wav")
                print(f"    Saved: sample_{i+1}_reconstructed.wav")
            
            # Store results
            results_summary.append({
                'sample': i+1,
                'mse': mse,
                'original_bpm': orig_bpm,
                'reconstructed_bpm': recon_bpm,
                'bpm_diff': bpm_diff,
                'beat_regularity_original': beat_score_orig,
                'beat_regularity_reconstructed': beat_score_recon
            })
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nReconstruction Quality:")
    print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    
    print(f"\nBPM Analysis:")
    print(f"  Average BPM difference: {np.mean(bpm_diffs):.2f} ± {np.std(bpm_diffs):.2f}")
    
    if beat_loss_scores:
        orig_beat_scores = [s[0] for s in beat_loss_scores]
        recon_beat_scores = [s[1] for s in beat_loss_scores]
        print(f"\nBeat Regularity (lower = weaker beats):")
        print(f"  Original: {np.mean(orig_beat_scores):.4f} ± {np.std(orig_beat_scores):.4f}")
        print(f"  Reconstructed: {np.mean(recon_beat_scores):.4f} ± {np.std(recon_beat_scores):.4f}")
    
    # Save detailed results
    import json
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': num_samples,
                'avg_mse': float(np.mean(mse_scores)),
                'std_mse': float(np.std(mse_scores)),
                'avg_bpm_diff': float(np.mean(bpm_diffs)),
                'std_bpm_diff': float(np.std(bpm_diffs)),
            },
            'samples': results_summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved: {results_path}")
    print(f"Visualizations saved in: {vis_dir}")
    if save_audio:
        print(f"Audio files saved in: {audio_dir}")
    print("="*70)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*70)
    print("VAE Model Evaluation")
    print("="*70)
    
    # Get device
    device = get_device(args.device)
    print(f"Device: {device}")
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Load model
    model = load_model(args.checkpoint, args.latent_dim, device)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    try:
        dataset = AudioMelDataset(
            data_dir=args.data_dir,
            sr=22050,
            duration=10.0,
            n_mels=128,
            hop_length=512,
            normalize=True
        )
        print(f"Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Check if soundfile is available (for audio saving)
    if args.save_audio:
        try:
            import soundfile
        except ImportError:
            print("Warning: soundfile not available. Cannot save audio files.")
            print("Install with: pip install soundfile")
            args.save_audio = False
    
    # Evaluate model
    output_dir = Path(args.output_dir)
    evaluate_model(
        model, dataset, args.num_samples, output_dir, device,
        save_audio=args.save_audio
    )
    
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
