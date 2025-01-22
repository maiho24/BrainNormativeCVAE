"""
Conditional Adversarial Autoencoder (CAAE) Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, non_linear=True):
        super().__init__()
        self.non_linear = non_linear
        
        # Create layer sizes list
        layer_sizes = [input_dim] + hidden_dim
        
        # Create hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out) 
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        
        # Output layer to latent space
        self.latent_layer = nn.Linear(hidden_dim[-1], latent_dim)
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            if self.non_linear:
                x = self.leaky_relu(x)
        return self.latent_layer(x)  # Linear activation for latent code

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, non_linear=True):
        """
        Args:
            latent_dim: Dimension of the latent space
            hidden_dim: List of hidden layer dimensions
            output_dim: Dimension of the output (brain features)
            condition_dim: Dimension of conditioning variables (includes all processed covariates:
                         continuous variables like Age and ICV, and categorical variables like
                         Sex, Diabetes Status, Smoking Status, etc.)
            non_linear: Whether to use non-linear activations between layers
        """
        super().__init__()
        self.non_linear = non_linear
        
        # Create layer sizes list (reversed hidden_dim)
        layer_sizes = [latent_dim + condition_dim] + hidden_dim[::-1]
        
        # Create hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out)
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim[0], output_dim)
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        for layer in self.hidden_layers:
            x = layer(x)
            if self.non_linear:
                x = self.leaky_relu(x)
        return self.output_layer(x)  # Linear activation for output

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, non_linear=True):
        super().__init__()
        self.non_linear = non_linear
        
        # Create layer sizes list
        layer_sizes = [input_dim] + hidden_dim
        
        # Create hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out)
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim[-1], 1)
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            if self.non_linear:
                x = self.leaky_relu(x)
        return torch.sigmoid(self.output_layer(x))

class AAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                condition_dim,
                learning_rate=0.0001,
                non_linear=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.non_linear = non_linear
        
        # Initialize components
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            non_linear=non_linear
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            condition_dim=condition_dim,
            non_linear=non_linear
        )
        
        self.discriminator = Discriminator(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            non_linear=non_linear
        )
        
        # Loss functions
        self.reconstruction_criterion = nn.MSELoss()
        self.adversarial_criterion = nn.BCELoss()
        
        # Initialize optimizers
        self.enc_dec_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, condition):
        return self.decoder(z, condition)

    def discriminate(self, z):
        return self.discriminator(z)

    def forward(self, x, condition):
        z = self.encode(x)
        return self.decode(z, condition), z

    def training_step(self, x, condition):
        """Perform one training step with adversarial and reconstruction updates."""
        batch_size = x.size(0)
        
        # 1. Autoencoder reconstruction phase
        z = self.encode(x)
        x_recon = self.decode(z, condition)
        recon_loss = self.reconstruction_criterion(x_recon, x)
        
        # 2. Discriminator phase
        real_z = torch.randn(batch_size, self.latent_dim, device=x.device)
        fake_z = z.detach()
        
        real_pred = self.discriminate(real_z)
        fake_pred = self.discriminate(fake_z)
        
        d_loss_real = self.adversarial_criterion(
            real_pred, torch.ones_like(real_pred))
        d_loss_fake = self.adversarial_criterion(
            fake_pred, torch.zeros_like(fake_pred))
        d_loss = d_loss_real + d_loss_fake
        
        # Update discriminator
        self.disc_optimizer.zero_grad()
        d_loss.backward()
        self.disc_optimizer.step()
        
        # 3. Generator (encoder) phase
        fake_pred = self.discriminate(z)
        g_loss = self.adversarial_criterion(
            fake_pred, torch.ones_like(fake_pred))
        
        # Update encoder and decoder
        self.enc_dec_optimizer.zero_grad()
        (recon_loss + g_loss).backward()
        self.enc_dec_optimizer.step()
        
        return {
            'reconstruction': recon_loss,
            'discriminator': d_loss,
            'generator': g_loss,
            'total_loss': recon_loss + g_loss
        }

    def compute_reconstruction_error(self, x, condition):
        """Compute reconstruction error for given samples."""
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
            x_recon = self.decode(z, condition)
            return F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
            