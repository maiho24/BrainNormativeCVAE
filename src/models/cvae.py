"""
Conditional Variational Autoencoder (cVAE) Implementation

This implementation extends the normative modelling framework by 
Lawry Aguila et al. (2022) (https://github.com/alawryaguila/normativecVAE),
with substantial refinements in both the model architecture and inference approach.

This implementation supports three variance modeling options:
  1. Shared decoder with two heads (fully learnable variance)
  2. Global self-learnable logvar (single learnable parameter)
  3. Covariate-specific logvar (small network)

References:
    Lawry Aguila, A., Chapman, J., Janahi, M., Altmann, A. (2022). 
    Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases. 
    GitHub Repository, https://github.com/alawryaguila/normativecVAE
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter
from enum import Enum
    
    
def compute_ll(Y, Y_recon):
    return -Y_recon.log_prob(Y).sum(1, keepdims=True).mean(0)

def compute_mse(Y, Y_recon):
    Y_recon_mean = Y_recon.loc
    return ((Y - Y_recon_mean)**2).mean()

def compute_huber_loss(Y, Y_recon, delta=1.0):
    Y_recon_mean = Y_recon.loc
    diff = Y - Y_recon_mean
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    return loss.mean()
    
    
class VarianceType(Enum):
    TWO_HEADS = "two_heads"
    GLOBAL_LEARNABLE = "global_learnable"
    COVARIATE_SPECIFIC = "covariate_specific"
    
    
class Encoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim, 
                c_dim,
                non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.c_dim = c_dim
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim + c_dim] + self.hidden_dims
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
               
        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)
        self.enc_logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)

    def forward(self, Y, c):
        c = c.reshape(-1, self.c_dim)
        conditional_Y = torch.cat((Y, c), dim=1)
        for it_layer, layer in enumerate(self.encoder_layers):
            conditional_Y = layer(conditional_Y)
            if self.non_linear:
                conditional_Y = F.relu(conditional_Y)

        mu_z = self.enc_mean_layer(conditional_Y)
        logvar_z = self.enc_logvar_layer(conditional_Y)
        return mu_z, logvar_z


class FlexibleDecoder(nn.Module):
    """
    Decoder with flexible variance modeling options.
    """
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                variance_type=VarianceType.TWO_HEADS,
                non_linear=True, 
                init_logvar=-0.5,
                variance_network_hidden_dim=[32]):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.variance_type = variance_type
        
        # Shared decoder layers
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim
        
        shared_layers = [nn.Linear(dim0, dim1, bias=True) 
                         for dim0, dim1 in zip(self.layer_sizes_decoder[:-2], 
                         self.layer_sizes_decoder[1:-1])]
        
        self.shared_decoder = nn.Sequential(*shared_layers)
        shared_output_dim = self.layer_sizes_decoder[-2]
        
        # Mean head (always present)
        self.mean_head = nn.Linear(shared_output_dim, input_dim, bias=True)
        
        # Variance modeling based on type
        ## Shared decoder network, then 2 heads
        if variance_type == VarianceType.TWO_HEADS:
            self.logvar_head = nn.Linear(shared_output_dim, input_dim, bias=True)
            # Initialize logvar head
            nn.init.constant_(self.logvar_head.bias, self.init_logvar)
            nn.init.xavier_uniform_(self.logvar_head.weight)
            
        elif variance_type == VarianceType.GLOBAL_LEARNABLE:
            # Single learnable parameter for all dimensions
            self.global_logvar = nn.Parameter(torch.full((input_dim,), self.init_logvar))
            
        elif variance_type == VarianceType.COVARIATE_SPECIFIC:
            # Small network that maps covariates to logvar
            variance_layers = []
            dims = [c_dim] + variance_network_hidden_dim + [input_dim]
            
            for i in range(len(dims) - 1):
                variance_layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
                if i < len(dims) - 2:
                    variance_layers.append(nn.ReLU())
            
            self.variance_network = nn.Sequential(*variance_layers)
            
            # Initialize the variance network
            final_layer = self.variance_network[-1]
            nn.init.constant_(final_layer.bias, self.init_logvar)
            nn.init.normal_(final_layer.weight, std=0.01)
        
    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        conditional_z = torch.cat((z, c), dim=1)
        
        # Process through shared decoder
        shared_features = conditional_z
        for layer in self.shared_decoder:
            shared_features = layer(shared_features)
            if self.non_linear:
                shared_features = F.relu(shared_features)
        
        # Get mean
        mu_out = self.mean_head(shared_features)
        
        # Get logvar based on variance type
        if self.variance_type == VarianceType.TWO_HEADS:
            logvar_out = self.logvar_head(shared_features)
            
        elif self.variance_type == VarianceType.GLOBAL_LEARNABLE:
            batch_size = mu_out.shape[0]
            logvar_out = self.global_logvar.unsqueeze(0).expand(batch_size, -1)
            
        elif self.variance_type == VarianceType.COVARIATE_SPECIFIC:
            logvar_out = self.variance_network(c)
        
        else:
            raise ValueError(f"Unknown variance_type: {self.variance_type}")
        
        # Stability checks
        if torch.isnan(mu_out).any():
            mu_out = torch.where(torch.isnan(mu_out), torch.zeros_like(mu_out), mu_out)
        # mu_out = torch.clamp(mu_out, min=-10.0, max=10.0)
        
        if torch.isnan(logvar_out).any():
            logvar_out = torch.where(torch.isnan(logvar_out), torch.full_like(logvar_out, self.init_logvar), logvar_out)
        
        # logvar_out = torch.clamp(logvar_out, min=-10.0, max=10.0)
        
        scale = torch.exp(0.5 * logvar_out) + 1e-6
        
        try:
            return Normal(loc=mu_out, scale=scale)
        except ValueError as e:
            print(f"Error creating Normal distribution: {e}")
            safe_mu = torch.zeros_like(mu_out)
            safe_scale = torch.ones_like(scale)
            return Normal(loc=safe_mu, scale=safe_scale)
            
class cVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                c_dim,
                beta=1,
                delta=1.0,
                non_linear=True,
                variance_type="two_heads",  # Can be "global_learnable", "covariate_specific" or "two_heads"
                variance_network_hidden_dim=[32],
                init_logvar=-0.5):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.non_linear=non_linear
        self.c_dim = c_dim
        self.beta = beta
        self.delta = delta
        
        if isinstance(variance_type, str):
            try:
                self.variance_type = VarianceType(variance_type)
            except ValueError:
                valid_types = [e.value for e in VarianceType]
                raise ValueError(f"Invalid variance_type: {variance_type}. Must be one of {valid_types}")
        else:
            self.variance_type = variance_type
        
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=self.non_linear)
        
        self.decoder = FlexibleDecoder(
            input_dim=input_dim, 
            hidden_dim=self.hidden_dim, 
            c_dim=c_dim,
            variance_type=self.variance_type,
            non_linear=self.non_linear,
            init_logvar=init_logvar,
            variance_network_hidden_dim=variance_network_hidden_dim
        )
    
    def encode(self, Y, c):
        return self.encoder(Y, c)

    def reparameterise(self, mu, logvar):
        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
            logvar = torch.where(torch.isnan(logvar), torch.zeros_like(logvar), logvar)
    
        # mu = torch.clamp(mu, min=-10.0, max=10.0)
        # logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        std = torch.exp(0.5 * logvar) + 1e-6
        if torch.isinf(std).any():
            std = torch.where(torch.isinf(std), torch.ones_like(std), std)
        
        eps = torch.randn_like(mu)
        z = mu + eps * std
        
        if torch.isnan(z).any():
            z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
            
        return z

    def decode(self, z, c):
        return self.decoder(z, c)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, Y, Y_recon):
        return compute_ll(Y, Y_recon)

    def calc_mse(self, Y, Y_recon):
        return compute_mse(Y, Y_recon)
        
    def calc_huber_loss(self, Y, Y_recon):
        return compute_huber_loss(Y, Y_recon, self.delta)
        
    def forward(self, Y, c):
        mu_z, logvar_z = self.encode(Y, c)
        z = self.reparameterise(mu_z, logvar_z)
        Y_recon = self.decode(z, c)
        fwd_rtn = {'Y_recon': Y_recon,
                   'mu_z': mu_z,
                   'logvar_z': logvar_z}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.rsample()

    def loss_function(self, Y, fwd_rtn):
        Y_recon = fwd_rtn['Y_recon']
        mu_z = fwd_rtn['mu_z']
        logvar_z = fwd_rtn['logvar_z']

        kl = self.calc_kl(mu_z, logvar_z)
        recon = self.calc_ll(Y, Y_recon)
        #var_reg = 0.01 * Y_recon.scale.abs().mean()
        
        total = self.beta*kl + recon
        losses = {'Total Loss': total,
                  'KL Divergence': kl,
                  'Reconstruction Loss': recon}
        return losses
    
    def get_variance_info(self, c=None, device=None):
        """
        Get variance information based on the variance type.
        For covariate-specific, requires covariates.
        """
        if self.variance_type == VarianceType.GLOBAL_LEARNABLE:
            with torch.no_grad():
                logvar = self.decoder.global_logvar
                variance = torch.exp(logvar)
                return {
                    'type': 'global',
                    'logvar': logvar.cpu().numpy(),
                    'variance': variance.cpu().numpy()
                }
                
        elif self.variance_type == VarianceType.COVARIATE_SPECIFIC:
            if c is None:
                raise ValueError("Covariates required for covariate-specific variance")
            c = torch.tensor(c, dtype=torch.float32).to(device)
            if len(c.shape) == 1:
                c = c.unsqueeze(0)
            
            with torch.no_grad():
                logvar = self.decoder.variance_network(c)
                variance = torch.exp(logvar)
                return {
                    'type': 'covariate_specific',
                    'logvar': logvar.cpu().numpy(),
                    'variance': variance.cpu().numpy()
                }
                
        else:
            return {
                'type': 'two_heads',
                'info': 'Variance is fully learnable through decoder head'
            }
            
    def pred_latent(self, Y, c, DEVICE):
        Y = torch.FloatTensor(Y.to_numpy()).to(DEVICE)
        c = torch.FloatTensor(c).to(DEVICE)
        with torch.no_grad():
            mu_z, logvar_z = self.encode(Y, c)   
        latent_mu = mu_z.cpu().detach().numpy()
        latent_var = logvar_z.exp().cpu().detach().numpy()
        return latent_mu, latent_var

    def pred_recon(self, Y, c,  DEVICE):
        Y = torch.FloatTensor(Y.to_numpy()).to(DEVICE)
        c = torch.FloatTensor(c).to(DEVICE)
        with torch.no_grad():
            mu_z, logvar_z = self.encode(Y, c)
            z = self.reparameterise(mu_z, logvar_z)
            Y_recon = self.decode(z, c)
            mu_Y_recon = Y_recon.loc.cpu().detach().numpy()
            var_Y_recon = Y_recon.scale.pow(2).cpu().detach().numpy()
        return mu_Y_recon, var_Y_recon