#!/usr/bin/env python3
"""
Training script for Mel-Spectrogram VAE
========================================
Complete training pipeline for the VAE model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import VAE model
from vae_model import MelSpectrogramVAE, vae_loss

# Import AudioMelDataset
try:
    from music_dataset.audio_dataset import AudioMelDataset
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'music_dataset'))
        from audio_dataset import AudioMelDataset
    except ImportError:
        print("Error: Could not import AudioMelDataset")
        print("Please ensure music_dataset/audio_dataset.py exists")
        sys.exit(1)


class VAETrainer:
    """Trainer class for VAE model."""
    
    def __init__(
        self,
        model: MelSpectrogramVAE,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        kl_weight: float = 1.0,
        device: torch.device = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        """Initialize VAE trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.kl_weight = kl_weight
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_div': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_div': [],
            'learning_rate': []
        }
        
        print(f"Trainer initialized on {self.device}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_div = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(data)
            loss, loss_dict = vae_loss(reconstruction, data, mu, logvar, self.kl_weight)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss_dict['total_loss']
            epoch_recon_loss += loss_dict['recon_loss']
            epoch_kl_div += loss_dict['kl_divergence']
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total_loss']:.4f}")
        
        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'kl_div': epoch_kl_div / num_batches
        }
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_loss = val_recon_loss = val_kl_div = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                reconstruction, mu, logvar = self.model(data)
                loss, loss_dict = vae_loss(reconstruction, data, mu, logvar, self.kl_weight)
                
                val_loss += loss_dict['total_loss']
                val_recon_loss += loss_dict['recon_loss']
                val_kl_div += loss_dict['kl_divergence']
                num_batches += 1
        
        return {
            'loss': val_loss / num_batches,
            'recon_loss': val_recon_loss / num_batches,
            'kl_div': val_kl_div / num_batches
        }
    
    def train(self, num_epochs: int, save_interval: int = 5):
        """Train the model."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print("="*70)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-"*70)
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_div'].append(train_metrics['kl_div'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_kl_div'].append(val_metrics['kl_div'])
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                self.scheduler.step(val_metrics['loss'])
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, 'best_model.pt')
                    print(f"  ✓ Best model saved!")
            
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
                self.generate_samples(epoch)
        
        print("\nTraining completed!")
        self.save_checkpoint(num_epochs, 'final_model.pt')
        self.save_history()
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def save_history(self):
        """Save training history."""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.history['train_recon_loss'], label='Train')
        if self.history['val_recon_loss']:
            axes[0, 1].plot(epochs, self.history['val_recon_loss'], label='Val')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, self.history['train_kl_div'], label='Train')
        if self.history['val_kl_div']:
            axes[1, 0].plot(epochs, self.history['val_kl_div'], label='Val')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Div')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, self.history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_curves.png', dpi=150)
        print(f"Training curves saved!")
        plt.close()
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save samples."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_samples):
            sample = samples[i, 0].cpu().numpy()
            axes[i].imshow(sample, aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / f'samples_epoch_{epoch}.png', dpi=150)
        plt.close()


def main():
    """Main training function."""
    print("="*70)
    print("VAE Training Script")
    print("="*70)
    
    # Configuration
    config = {
        'data_dir': './weak_beat_music',
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'latent_dim': 128,
        'kl_weight': 1.0,
        'save_interval': 5,
        'num_workers': 2
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check data directory
    if not Path(config['data_dir']).exists():
        print(f"\nError: Data directory '{config['data_dir']}' not found!")
        print("Please filter audio data first using filter_fma_weak_beat/fma_filter.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    try:
        full_dataset = AudioMelDataset(
            data_dir=config['data_dir'],
            sr=22050,
            duration=10.0,
            n_mels=128,
            hop_length=512
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = MelSpectrogramVAE(latent_dim=config['latent_dim'])
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        kl_weight=config['kl_weight'],
        device=device
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_interval=config['save_interval'])
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
