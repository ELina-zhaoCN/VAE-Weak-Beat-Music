#!/usr/bin/env python3
"""
Complete Training Script for Weak Beat Music VAE
================================================
Trains a VAE to generate music with weak/irregular beats using:
- Reconstruction loss (MSE)
- KL divergence loss
- Beat regularity loss
- Beta-VAE warm-up strategy
- BPM accuracy evaluation

Usage:
    python train.py --data_dir ./weak_beat_music --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import librosa

# Try to import tensorboard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

# Import custom modules
try:
    # Try to import from organized structure
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
            print("Make sure music_dataset/audio_dataset.py exists")
            sys.exit(1)
    
    # Import beat loss
    try:
        from beat_loss import beat_loss, BeatLoss
    except ImportError:
        print("Warning: beat_loss not found. Beat loss will not be used.")
        beat_loss = None
        BeatLoss = None

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VAE for weak beat music generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./weak_beat_music',
                       help='Path to music dataset directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation set ratio (0.0-1.0)')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent space dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Loss weights
    parser.add_argument('--kl_weight', type=float, default=1.0,
                       help='Initial KL divergence weight (beta)')
    parser.add_argument('--beat_weight', type=float, default=0.1,
                       help='Initial beat loss weight (gamma)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Number of epochs for warmup')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Evaluation
    parser.add_argument('--eval_bpm', action='store_true',
                       help='Evaluate BPM accuracy (slow)')
    parser.add_argument('--no_beat_loss', action='store_true',
                       help='Disable beat loss')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: librosa uses numpy's random state


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create training and validation data loaders.
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_samples: Total number of samples
    """
    print(f"\nLoading dataset from: {args.data_dir}")
    
    # Check if directory exists
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Please run filter_fma_weak_beat/fma_filter.py first to create the dataset."
        )
    
    # Create full dataset
    try:
        full_dataset = AudioMelDataset(
            data_dir=args.data_dir,
            sr=22050,
            duration=10.0,
            n_mels=128,
            hop_length=512,
            normalize=True
        )
    except Exception as e:
        raise RuntimeError(f"Error creating dataset: {e}")
    
    num_samples = len(full_dataset)
    print(f"Total samples: {num_samples}")
    
    # Split into train/val
    val_size = int(args.val_split * num_samples)
    train_size = num_samples - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, num_samples


def get_warmup_weights(epoch: int, warmup_epochs: int, 
                      initial_weight: float) -> float:
    """
    Get warmup weight using linear schedule.
    
    Weight increases linearly from 0 to initial_weight over warmup_epochs,
    then stays constant.
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs
        initial_weight: Target weight after warmup
    
    Returns:
        weight: Current weight
    """
    if epoch >= warmup_epochs:
        return initial_weight
    else:
        return initial_weight * (epoch / warmup_epochs)


def compute_vae_loss(reconstruction: torch.Tensor,
                     target: torch.Tensor,
                     mu: torch.Tensor,
                     logvar: torch.Tensor,
                     kl_weight: float = 1.0) -> Tuple[torch.Tensor, Dict]:
    """
    Compute VAE loss: Reconstruction + KL divergence.
    
    Args:
        reconstruction: Reconstructed output (batch, 1, 128, 431)
        target: Original input (batch, 1, 128, 431)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log variance (batch, latent_dim)
        kl_weight: Weight for KL divergence (beta)
    
    Returns:
        total_loss: Total VAE loss
        loss_dict: Dictionary with loss components
    """
    batch_size = target.size(0)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    recon_loss = recon_loss / batch_size
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence = kl_divergence / batch_size
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_divergence
    
    loss_dict = {
        'recon': recon_loss.item(),
        'kl': kl_divergence.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def estimate_bpm_from_spectrogram(mel_spec: torch.Tensor,
                                  sr: int = 22050,
                                  hop_length: int = 512) -> float:
    """
    Estimate BPM from mel-spectrogram using autocorrelation.
    
    This is used as a proxy when ground truth BPM is unavailable.
    
    Args:
        mel_spec: Mel-spectrogram (1, n_mels, n_frames) or (n_mels, n_frames)
        sr: Sampling rate
        hop_length: Hop length used for spectrogram
    
    Returns:
        bpm: Estimated BPM
    """
    # Convert to numpy and get energy envelope
    if mel_spec.dim() == 3:
        mel_spec = mel_spec[0]  # Remove channel dimension
    
    mel_np = mel_spec.cpu().numpy()
    
    # Average over frequency to get energy envelope
    energy = np.mean(mel_np[:10], axis=0)  # Use low frequencies
    
    # Compute autocorrelation
    autocorr = np.correlate(energy, energy, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    # Find peaks (exclude first few lags)
    min_lag = int(0.3 * sr / hop_length)  # Minimum ~0.3 seconds
    max_lag = int(2.0 * sr / hop_length)  # Maximum ~2 seconds
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    
    # Find first significant peak
    search_range = autocorr[min_lag:max_lag]
    if len(search_range) == 0:
        return 0.0
    
    peak_lag = np.argmax(search_range) + min_lag
    
    # Convert lag to BPM
    tempo_hz = sr / (hop_length * peak_lag)
    bpm = tempo_hz * 60
    
    return float(bpm)


def evaluate_bpm_accuracy(model: nn.Module,
                         dataloader: DataLoader,
                         device: torch.device,
                         max_batches: int = 10) -> Dict:
    """
    Evaluate BPM accuracy on validation set.
    
    Uses autocorrelation consistency as a proxy for BPM accuracy
    since we don't have ground truth BPM labels.
    
    Args:
        model: VAE model
        dataloader: Validation data loader
        device: Device to use
        max_batches: Maximum number of batches to evaluate (for speed)
    
    Returns:
        metrics: Dictionary with BPM metrics
    """
    model.eval()
    
    bpm_original = []
    bpm_reconstructed = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            
            # Estimate BPM for each sample
            for i in range(batch.size(0)):
                bpm_orig = estimate_bpm_from_spectrogram(batch[i])
                bpm_recon = estimate_bpm_from_spectrogram(reconstruction[i])
                
                # Only include valid estimates
                if bpm_orig > 0 and bpm_recon > 0:
                    bpm_original.append(bpm_orig)
                    bpm_reconstructed.append(bpm_recon)
    
    if len(bpm_original) == 0:
        return {
            'bpm_mae': 0.0,
            'bpm_consistency': 0.0,
            'avg_bpm_original': 0.0,
            'avg_bpm_reconstructed': 0.0
        }
    
    bpm_original = np.array(bpm_original)
    bpm_reconstructed = np.array(bpm_reconstructed)
    
    # Mean absolute error
    mae = np.mean(np.abs(bpm_original - bpm_reconstructed))
    
    # Consistency: how similar are the BPMs
    consistency = 1.0 - np.clip(mae / 100.0, 0, 1)  # Normalize to [0, 1]
    
    metrics = {
        'bpm_mae': float(mae),
        'bpm_consistency': float(consistency),
        'avg_bpm_original': float(np.mean(bpm_original)),
        'avg_bpm_reconstructed': float(np.mean(bpm_reconstructed))
    }
    
    return metrics


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                args,
                writer: Optional[SummaryWriter] = None) -> Dict:
    """
    Train for one epoch.
    
    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number (0-indexed)
        args: Command-line arguments
        writer: Tensorboard writer (optional)
    
    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    
    # Get warmup weights
    kl_weight = get_warmup_weights(epoch, args.warmup_epochs, args.kl_weight)
    beat_weight = get_warmup_weights(epoch, args.warmup_epochs, args.beat_weight)
    
    # Initialize metrics
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'beat': 0.0
    }
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Forward pass
        reconstruction, mu, logvar = model(batch)
        
        # Compute VAE loss (reconstruction + KL)
        vae_loss, loss_dict = compute_vae_loss(
            reconstruction, batch, mu, logvar, kl_weight
        )
        
        # Compute beat loss (if enabled)
        if not args.no_beat_loss and beat_loss is not None:
            beat_loss_val = beat_loss(reconstruction, normalize=True)
            beat_loss_weighted = beat_weight * beat_loss_val
        else:
            beat_loss_val = torch.tensor(0.0)
            beat_loss_weighted = torch.tensor(0.0)
        
        # Total loss
        total_loss = vae_loss + beat_loss_weighted
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        epoch_losses['total'] += total_loss.item()
        epoch_losses['recon'] += loss_dict['recon']
        epoch_losses['kl'] += loss_dict['kl']
        epoch_losses['beat'] += beat_loss_val.item() if isinstance(beat_loss_val, torch.Tensor) else 0.0
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'recon': f"{loss_dict['recon']:.4f}",
            'kl': f"{loss_dict['kl']:.4f}",
            'beat': f"{beat_loss_val.item():.4f}" if isinstance(beat_loss_val, torch.Tensor) else "0.0000",
            'β': f"{kl_weight:.3f}",
            'γ': f"{beat_weight:.3f}"
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('Train/recon_loss', loss_dict['recon'], global_step)
            writer.add_scalar('Train/kl_loss', loss_dict['kl'], global_step)
            if isinstance(beat_loss_val, torch.Tensor):
                writer.add_scalar('Train/beat_loss', beat_loss_val.item(), global_step)
            writer.add_scalar('Train/kl_weight', kl_weight, global_step)
            writer.add_scalar('Train/beat_weight', beat_weight, global_step)
    
    # Average metrics
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model: nn.Module,
            dataloader: DataLoader,
            device: torch.device,
            args) -> Dict:
    """
    Validate the model.
    
    Args:
        model: VAE model
        dataloader: Validation data loader
        device: Device to use
        args: Command-line arguments
    
    Returns:
        metrics: Dictionary with validation metrics
    """
    model.eval()
    
    val_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'beat': 0.0
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            batch = batch.to(device)
            
            # Forward pass
            reconstruction, mu, logvar = model(batch)
            
            # Compute VAE loss
            vae_loss, loss_dict = compute_vae_loss(
                reconstruction, batch, mu, logvar, args.kl_weight
            )
            
            # Compute beat loss
            if not args.no_beat_loss and beat_loss is not None:
                beat_loss_val = beat_loss(reconstruction, normalize=True)
            else:
                beat_loss_val = torch.tensor(0.0)
            
            # Total loss
            total_loss = vae_loss + args.beat_weight * beat_loss_val
            
            # Update metrics
            val_losses['total'] += total_loss.item()
            val_losses['recon'] += loss_dict['recon']
            val_losses['kl'] += loss_dict['kl']
            val_losses['beat'] += beat_loss_val.item() if isinstance(beat_loss_val, torch.Tensor) else 0.0
            num_batches += 1
    
    # Average metrics
    for key in val_losses:
        val_losses[key] /= num_batches
    
    return val_losses


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   args,
                   filename: str = 'checkpoint.pt'):
    """Save model checkpoint."""
    checkpoint_path = Path(args.checkpoint_dir) / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   checkpoint_path: str,
                   device: torch.device) -> int:
    """
    Load model checkpoint.
    
    Returns:
        start_epoch: Epoch to resume from
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")
    
    return start_epoch


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"\n{'='*70}")
    print("Training Weak Beat Music VAE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = SummaryWriter(log_dir)
        print(f"Tensorboard log dir: {log_dir}")
    
    # Create data loaders
    train_loader, val_loader, num_samples = create_dataloaders(args)
    
    # Create model
    print(f"\nCreating model...")
    model = MelSpectrogramVAE(latent_dim=args.latent_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Save configuration
    config_path = Path(args.checkpoint_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved: {config_path}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_beat_loss': [],
        'best_epoch': 0,
        'best_val_beat_loss': float('inf')
    }
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    best_val_beat_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args, writer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        # Log epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train - Total: {train_metrics['total']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, "
              f"KL: {train_metrics['kl']:.4f}, "
              f"Beat: {train_metrics['beat']:.4f}")
        print(f"  Val   - Total: {val_metrics['total']:.4f}, "
              f"Recon: {val_metrics['recon']:.4f}, "
              f"KL: {val_metrics['kl']:.4f}, "
              f"Beat: {val_metrics['beat']:.4f}")
        
        # Evaluate BPM accuracy (if enabled)
        if args.eval_bpm and (epoch + 1) % 5 == 0:
            print("  Evaluating BPM accuracy...")
            bpm_metrics = evaluate_bpm_accuracy(model, val_loader, device, max_batches=5)
            print(f"  BPM MAE: {bpm_metrics['bpm_mae']:.2f}, "
                  f"Consistency: {bpm_metrics['bpm_consistency']:.4f}")
            
            if writer is not None:
                writer.add_scalar('Val/bpm_mae', bpm_metrics['bpm_mae'], epoch)
                writer.add_scalar('Val/bpm_consistency', bpm_metrics['bpm_consistency'], epoch)
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Epoch/train_loss', train_metrics['total'], epoch)
            writer.add_scalar('Epoch/val_loss', val_metrics['total'], epoch)
            writer.add_scalar('Epoch/val_beat_loss', val_metrics['beat'], epoch)
        
        # Update history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['val_beat_loss'].append(val_metrics['beat'])
        
        # Save best model (lowest validation beat loss)
        if val_metrics['beat'] < best_val_beat_loss:
            best_val_beat_loss = val_metrics['beat']
            history['best_epoch'] = epoch
            history['best_val_beat_loss'] = best_val_beat_loss
            save_checkpoint(model, optimizer, epoch, val_metrics, args, 'best_model.pt')
            print(f"  ✓ New best model! Val beat loss: {best_val_beat_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, args, 
                          f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save history
        history_path = Path(args.checkpoint_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs-1, val_metrics, args, 'final_model.pt')
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    # Print final summary
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Best validation beat loss: {history['best_val_beat_loss']:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    if TENSORBOARD_AVAILABLE:
        print(f"Tensorboard logs: {args.log_dir}")
        print(f"Run: tensorboard --logdir={args.log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
