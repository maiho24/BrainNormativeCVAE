"""
Conditional Variational Autoencoder (cVAE) Implementation

This implementation extends the normative modelling framework by 
Lawry Aguila et al. (2022) (https://github.com/alawryaguila/normativecVAE),
with refinements in both the model architecture and inference approach.

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


def compute_ll(Y, Y_recon):
    return -Y_recon.log_prob(Y).sum(1, keepdims=True).mean(0)

def compute_mse(Y, Y_recon):
    Y_recon_mean = Y_recon.loc
    return ((Y - Y_recon_mean)**2).mean()


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


class Decoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                non_linear=False, 
                init_logvar=-3.0):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:])]
        
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(self.layer_sizes_decoder[-2],self.layer_sizes_decoder[-1], bias=True)
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)


    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        conditional_z = torch.cat((z, c), dim=1)
        for it_layer, layer in enumerate(self.decoder_layers):
            conditional_z = layer(conditional_z)
            if self.non_linear:
                conditional_z = F.relu(conditional_z)

        mu_out = self.decoder_mean_layer(conditional_z) 
        logvar_out = torch.clamp(self.logvar_out, min=-10.0, max=10.0)
        scale = torch.exp(0.5 * logvar_out).expand_as(mu_out)
        scale = scale + 1e-6
        mu_out = torch.clamp(mu_out, min=-1e6, max=1e6)
        return Normal(loc=mu_out, scale=scale)


class cVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.001,
                beta=1,
                non_linear=False):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.beta = beta
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
        self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate) 
    
    def encode(self, Y, c):
        return self.encoder(Y, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c):
        return self.decoder(z, c)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, Y, Y_recon):
        return compute_ll(Y, Y_recon)

    def calc_mse(self, Y, Y_recon):
        return compute_mse(Y, Y_recon)
        
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

        total = self.beta*kl + recon
        losses = {'Total Loss': total,
                  'KL Divergence': kl,
                  'Reconstruction Loss': recon}
        return losses

    def pred_latent(self, Y, c, DEVICE):
        Y = torch.FloatTensor(Y.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu_z, logvar_z = self.encode(Y, c)   
        latent_mu = mu_z.cpu().detach().numpy()
        latent_var = logvar_z.exp().cpu().detach().numpy()
        return latent_mu, latent_var

    def pred_recon(self, Y, c,  DEVICE):
        Y = torch.FloatTensor(Y.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu_z, logvar_z = self.encode(Y, c)
            z = self.reparameterise(mu_z, logvar_z)
            Y_recon = self.decode(z, c)
            mu_Y_recon = Y_recon.loc.cpu().detach().numpy()
            var_Y_recon = Y_recon.scale.pow(2).cpu().detach().numpy()
        return mu_Y_recon, var_Y_recon