import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualQuantizer(nn.Module):
    """
    One level of residual quantization: maps residuals to nearest codebook embeddings.
    """
    def __init__(self, embedding_dim: int, codebook_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)

    def forward(self, residual: torch.Tensor) -> (torch.Tensor, torch.LongTensor):
        # residual: (batch, embedding_dim)
        # Compute distances to codebook entries
        # embedding: (K, D)
        emb = self.codebook.weight  # (K, D)
        # Expand for vectorized distance computation
        # residual.unsqueeze(1): (batch, 1, D), emb.unsqueeze(0): (1, K, D)
        dists = torch.sum((residual.unsqueeze(1) - emb.unsqueeze(0))**2, dim=2)  # (batch, K)
        codes = torch.argmin(dists, dim=1)  # (batch,)
        quantized = F.embedding(codes, emb)  # (batch, D)
        return quantized, codes

class RQVAE(nn.Module):
    """
    Residual-Quantized VAE with m quantization levels.
    Encodes input x to latent z, then quantizes z into m codewords.
    Decoder reconstructs input from quantized representation.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 latent_dim: int,
                 num_quantizers: int,
                 codebook_size: int,
                 commitment_cost: float = 0.25):
        super().__init__()
        # Encoder network: produce latent mean and logvar
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(last_dim, h_dim))
            modules.append(nn.ReLU())
            last_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # Quantizers
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            ResidualQuantizer(latent_dim, codebook_size)
            for _ in range(num_quantizers)
        ])
        self.commitment_cost = commitment_cost

        # Decoder network: reconstruct input
        modules = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            modules.append(nn.Linear(last_dim, h_dim))
            modules.append(nn.ReLU())
            last_dim = h_dim
        modules.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Return latent mean and logvar"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
      """
      "Reparameterization trick” used in VAEs to turn sampling—which is non-differentiable—into 
      a differentiable operation so you can backpropagate through it
      """
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

    def quantize(self, z: torch.Tensor) -> (torch.Tensor, torch.LongTensor):
        """
        Apply multi-level residual quantization:
        z: (batch, latent_dim)
        returns quantized_sum: (batch, latent_dim), codes: (batch, num_quantizers)
        """
        residual = z
        quantized_sum = 0
        codes = []
        for q in self.quantizers:
            q_out, code = q(residual)
            quantized_sum = quantized_sum + q_out
            residual = residual - q_out
            codes.append(code)
        codes = torch.stack(codes, dim=1)  # (batch, m)
        return quantized_sum, codes

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> dict:
        # VAE encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Residual quantization
        quantized, codes = self.quantize(z)
        # Decode
        x_recon = self.decode(quantized)
        return {
            'recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'quantized': quantized,
            'codes': codes
        }

    def loss_function(self, x: torch.Tensor, outputs: dict) -> torch.Tensor:
        # Reconstruction loss (e.g., MSE)
        recon_loss = F.mse_loss(outputs['recon'], x, reduction='mean')
        # VAE KL term
        mu, logvar = outputs['mu'], outputs['logvar']
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / x.size(0)
        # Commitment loss: ||sg(z_e) - e||^2
        commitment_loss = 0
        residual = self.reparameterize(mu, logvar)
        quantized_sum = 0
        for i, q in enumerate(self.quantizers):
            q_out, _ = q(residual)
            commitment_loss += F.mse_loss(residual.detach(), q_out)
            residual = residual - q_out
        commitment_loss = self.commitment_cost * commitment_loss
        return recon_loss + kld + commitment_loss

# Example instantiation:
# model = RQVAE(input_dim=768, hidden_dims=[512,256], latent_dim=128,
#               num_quantizers=4, codebook_size=256)
