"""
Convolutional Variational Autoencoder (VAE) for Mel-Spectrograms
=================================================================
PyTorch implementation of a VAE for audio generation using Mel-spectrograms.
Designed for input shape: (batch_size, 1, 128, 431)

Architecture:
- Encoder: Convolutional layers → Latent representation (mu, logvar)
- Decoder: Transposed convolutional layers → Reconstructed spectrogram
- Latent dimension: 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


class MelSpectrogramVAE(nn.Module):
    """
    Variational Autoencoder for Mel-Spectrogram generation.
    
    Input shape: (batch_size, 1, 128, 431)
    Latent dimension: 128
    
    Architecture Details:
    --------------------
    Encoder:
        Conv2d(1, 32, 4, 2, 1)   -> BatchNorm -> ReLU  : (32, 64, 215)
        Conv2d(32, 64, 4, 2, 1)  -> BatchNorm -> ReLU  : (64, 32, 107)
        Conv2d(64, 128, 4, 2, 1) -> BatchNorm -> ReLU  : (128, 16, 53)
        Flatten -> (128 * 16 * 53 = 108,544)
        Linear -> mu(128), logvar(128)
    
    Decoder:
        Linear(128, 108544)
        Reshape -> (128, 16, 53)
        ConvTranspose2d(128, 64, 4, 2, 1) -> BatchNorm -> ReLU : (64, 32, 106)
        ConvTranspose2d(64, 32, 4, 2, 1)  -> BatchNorm -> ReLU : (32, 64, 212)
        ConvTranspose2d(32, 1, 4, 2, 1)                        : (1, 128, 424)
        Final adjustment to (1, 128, 431)
    """
    
    def __init__(self, latent_dim: int = 128, input_channels: int = 1):
        """
        Initialize the VAE.
        
        Args:
            latent_dim: Dimension of the latent space (default: 128)
            input_channels: Number of input channels (default: 1)
        """
        super(MelSpectrogramVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        # Input dimensions
        self.input_height = 128
        self.input_width = 431
        
        # Calculate dimensions after each convolution
        # Formula: out = (in + 2*padding - kernel_size) // stride + 1
        
        # After conv1: (128, 431) -> (64, 215)
        h1 = (128 + 2*1 - 4) // 2 + 1  # = 64
        w1 = (431 + 2*1 - 4) // 2 + 1  # = 215
        
        # After conv2: (64, 215) -> (32, 107)
        h2 = (h1 + 2*1 - 4) // 2 + 1   # = 32
        w2 = (w1 + 2*1 - 4) // 2 + 1   # = 107
        
        # After conv3: (32, 107) -> (16, 53)
        h3 = (h2 + 2*1 - 4) // 2 + 1   # = 16
        w3 = (w2 + 2*1 - 4) // 2 + 1   # = 53
        
        self.conv_output_height = h3
        self.conv_output_width = w3
        self.conv_output_channels = 128
        
        # Flattened dimension: 128 * 16 * 53 = 108,544
        self.flatten_dim = self.conv_output_channels * h3 * w3
        
        print(f"VAE Architecture:")
        print(f"  Input: (batch, {input_channels}, {self.input_height}, {self.input_width})")
        print(f"  After conv layers: (batch, {self.conv_output_channels}, {h3}, {w3})")
        print(f"  Flattened dimension: {self.flatten_dim}")
        print(f"  Latent dimension: {latent_dim}")
        
        # ============= ENCODER =============
        self.encoder = nn.Sequential(
            # Conv1: (1, 128, 431) -> (32, 64, 215)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: (32, 64, 215) -> (64, 32, 107)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: (64, 32, 107) -> (128, 16, 53)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Linear layers to latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # ============= DECODER =============
        # Linear layer to expand from latent space
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            # ConvTranspose1: (128, 16, 53) -> (64, 32, 106)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # ConvTranspose2: (64, 32, 106) -> (32, 64, 212)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # ConvTranspose3: (32, 64, 212) -> (1, 128, 424)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )
        
        # Final adjustment layer to get exact output size (128, 431)
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 431)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Pass through encoder convolutions
        x = self.encoder(x)  # (batch, 128, 16, 53)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)  # (batch, 108544)
        
        # Get mu and logvar
        mu = self.fc_mu(x)  # (batch, latent_dim)
        logvar = self.fc_logvar(x)  # (batch, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise from N(0,1)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed spectrogram.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            reconstruction: Reconstructed spectrogram (batch_size, 1, 128, 431)
        """
        # Expand latent vector
        x = self.fc_decode(z)  # (batch, 108544)
        
        # Reshape to feature maps
        x = x.view(-1, self.conv_output_channels, 
                   self.conv_output_height, 
                   self.conv_output_width)  # (batch, 128, 16, 53)
        
        # Pass through decoder
        x = self.decoder(x)  # (batch, 1, 128, 424)
        
        # Apply final convolution
        x = self.final_conv(x)  # (batch, 1, 128, 424)
        
        # Pad width from 424 to 431 (add 7 columns to the right)
        x = F.pad(x, (0, 7, 0, 0), mode='constant', value=0)  # (batch, 1, 128, 431)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input spectrogram (batch_size, 1, 128, 431)
            
        Returns:
            reconstruction: Reconstructed spectrogram (batch_size, 1, 128, 431)
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize (sample from latent distribution)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by sampling from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated spectrograms (num_samples, 1, 128, 431)
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode
        samples = self.decode(z)
        
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input (deterministic, using mean of latent distribution).
        
        Args:
            x: Input spectrogram (batch_size, 1, 128, 431)
            
        Returns:
            reconstruction: Reconstructed spectrogram (batch_size, 1, 128, 431)
        """
        mu, _ = self.encode(x)
        reconstruction = self.decode(mu)
        return reconstruction


def vae_loss(reconstruction: torch.Tensor, 
             target: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor,
             kl_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    VAE loss function: Reconstruction loss + KL divergence.
    
    Args:
        reconstruction: Reconstructed output (batch_size, 1, 128, 431)
        target: Original input (batch_size, 1, 128, 431)
        mu: Mean of latent distribution (batch_size, latent_dim)
        logvar: Log variance of latent distribution (batch_size, latent_dim)
        kl_weight: Weight for KL divergence term (default: 1.0)
        
    Returns:
        total_loss: Total VAE loss
        loss_dict: Dictionary with individual loss components
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_divergence
    
    # Average over batch
    batch_size = target.size(0)
    total_loss = total_loss / batch_size
    recon_loss = recon_loss / batch_size
    kl_divergence = kl_divergence / batch_size
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_divergence': kl_divergence.item()
    }
    
    return total_loss, loss_dict


def test_vae():
    """Test function for the VAE model."""
    print("="*70)
    print("Testing MelSpectrogramVAE")
    print("="*70)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = MelSpectrogramVAE(latent_dim=128).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n" + "-"*70)
    print("Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 431).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    reconstruction, mu, logvar = model(x)
    
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Verify shapes
    assert reconstruction.shape == x.shape, f"Output shape mismatch! Expected {x.shape}, got {reconstruction.shape}"
    assert mu.shape == (batch_size, 128), f"Mu shape mismatch! Expected ({batch_size}, 128), got {mu.shape}"
    assert logvar.shape == (batch_size, 128), f"Logvar shape mismatch! Expected ({batch_size}, 128), got {logvar.shape}"
    print("✓ All shapes correct!")
    
    # Test loss
    print("\n" + "-"*70)
    print("Testing loss function...")
    loss, loss_dict = vae_loss(reconstruction, x, mu, logvar)
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss']:.4f}")
    print(f"KL divergence: {loss_dict['kl_divergence']:.4f}")
    
    # Test sampling
    print("\n" + "-"*70)
    print("Testing sampling...")
    num_samples = 8
    samples = model.sample(num_samples, device)
    print(f"Generated {num_samples} samples")
    print(f"Samples shape: {samples.shape}")
    assert samples.shape == (num_samples, 1, 128, 431), "Sample shape mismatch!"
    print("✓ Sampling works!")
    
    # Test reconstruction (deterministic)
    print("\n" + "-"*70)
    print("Testing deterministic reconstruction...")
    recon = model.reconstruct(x)
    print(f"Reconstruction shape: {recon.shape}")
    assert recon.shape == x.shape, "Reconstruction shape mismatch!"
    print("✓ Reconstruction works!")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    test_vae()
