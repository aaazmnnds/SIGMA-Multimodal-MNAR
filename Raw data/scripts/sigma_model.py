# SIGMA model implementation. Architecture follows the manuscript specification exactly.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SIGMADataset(Dataset):
    def __init__(self, x_c, x_g, T, E, mask_c, mask_g):
        self.x_c = torch.tensor(x_c, dtype=torch.float32)
        self.x_g = torch.tensor(x_g, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)
        self.mask_c = torch.tensor(mask_c, dtype=torch.float32)
        self.mask_g = torch.tensor(mask_g, dtype=torch.float32)

    def __len__(self):
        return self.x_c.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_c[idx],
            self.mask_c[idx],
            self.x_g[idx],
            self.mask_g[idx],
            self.T[idx],
            self.E[idx]
        )


class Encoder(nn.Module):
    def __init__(self, input_dim, d):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, d)
        self.mu_head = nn.Linear(d, d)
        self.log_var_head = nn.Linear(d, d)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


class EarlyFusion(nn.Module):
    def __init__(self, d):
        super(EarlyFusion, self).__init__()
        self.proj = nn.Linear(2 * d, d)

    def forward(self, z_c, z_g):
        z_cat = torch.cat([z_c, z_g], dim=-1)
        z_shared = self.proj(z_cat)
        return z_shared


class CrossModalAttention(nn.Module):
    def __init__(self, d):
        super(CrossModalAttention, self).__init__()
        self.attn_c2g = nn.MultiheadAttention(embed_dim=d, num_heads=1, batch_first=True)
        self.attn_g2c = nn.MultiheadAttention(embed_dim=d, num_heads=1, batch_first=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, z_c, z_g):
        z_c_seq = z_c.unsqueeze(1)
        z_g_seq = z_g.unsqueeze(1)
        
        # clinical queries genomic
        attn_c, _ = self.attn_c2g(query=z_c_seq, key=z_g_seq, value=z_g_seq)
        # genomic queries clinical
        attn_g, _ = self.attn_g2c(query=z_g_seq, key=z_c_seq, value=z_c_seq)
        
        z_c_att = attn_c.squeeze(1)
        z_g_att = attn_g.squeeze(1)
        
        z_cat = torch.cat([z_c_att, z_g_att], dim=-1)
        z_shared = self.proj(z_cat)
        return z_shared


class ProductOfExperts(nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()

    def forward(self, mu_c, log_var_c, mu_g, log_var_g):
        var_c = torch.exp(log_var_c) + 1e-8
        var_g = torch.exp(log_var_g) + 1e-8
        
        var_shared = 1.0 / (1.0 / var_c + 1.0 / var_g)
        mu_shared = var_shared * (mu_c / var_c + mu_g / var_g)
        log_var_shared = torch.log(var_shared)
        log_var_shared = torch.clamp(log_var_shared, min=-10.0, max=10.0)
        
        return mu_shared, log_var_shared


class Decoder(nn.Module):
    def __init__(self, d, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(d, d)
        self.out_head = nn.Linear(d, output_dim)

    def forward(self, z_shared):
        h = F.relu(self.fc1(z_shared))
        return self.out_head(h)


class SurvivalNet(nn.Module):
    def __init__(self, d):
        super(SurvivalNet, self).__init__()
        self.fc1 = nn.Linear(d, d)
        self.out_head = nn.Linear(d, 1)

    def forward(self, z_shared):
        h = F.relu(self.fc1(z_shared))
        eta = self.out_head(h).squeeze(-1)
        return eta


class SIGMA(nn.Module):
    def __init__(self, p_c, p_g, d, fusion_strategy):
        super(SIGMA, self).__init__()
        self.p_c = p_c
        self.p_g = p_g
        self.d = d
        self.fusion_strategy = fusion_strategy

        self.enc_c = Encoder(2 * p_c, d)
        self.enc_g = Encoder(2 * p_g, d)

        if fusion_strategy == 'EF':
            self.fusion_module = EarlyFusion(d)
        elif fusion_strategy == 'CMA':
            self.fusion_module = CrossModalAttention(d)
        elif fusion_strategy == 'PoE':
            self.fusion_module = ProductOfExperts()
        else:
            raise ValueError("fusion_strategy must be 'EF', 'CMA', or 'PoE'")

        self.dec_c = Decoder(d, p_c)
        self.dec_g = Decoder(d, p_g)
        self.survival_net = SurvivalNet(d)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x_c, mask_c, x_g, mask_g):
        in_c = torch.cat([x_c, mask_c], dim=-1)
        in_g = torch.cat([x_g, mask_g], dim=-1)

        mu_c, log_var_c = self.enc_c(in_c)
        mu_g, log_var_g = self.enc_g(in_g)

        z_c = self.reparameterize(mu_c, log_var_c)
        z_g = self.reparameterize(mu_g, log_var_g)

        if self.fusion_strategy == 'PoE':
            mu_shared, log_var_shared = self.fusion_module(mu_c, log_var_c, mu_g, log_var_g)
            z_shared = self.reparameterize(mu_shared, log_var_shared)
        else:
            z_shared = self.fusion_module(z_c, z_g)
            mu_shared = (mu_c + mu_g) / 2.0
            log_var_shared = (log_var_c + log_var_g) / 2.0

        x_hat_c = self.dec_c(z_shared)
        x_hat_g = self.dec_g(z_shared)
        eta = self.survival_net(z_shared)

        return x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared


def masked_mse(x_hat, x_true, mask):
    observed = (mask == 0).float()
    loss = ((x_hat - x_true) ** 2) * observed
    return loss.sum() / (observed.sum() + 1e-8)


def kl_divergence(mu, log_var):
    # Sum over the latent dimension (dim=1), then mean over the batch (dim=0)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl.mean()


def cox_partial_likelihood(eta, T, E):
    order = torch.argsort(T, descending=True)
    eta = eta[order]
    E = E[order]
    log_cumsum_exp = torch.logcumsumexp(eta, dim=0)
    uncensored_loss = (eta - log_cumsum_exp) * E
    n_events = E.sum()
    return -uncensored_loss.sum() / (n_events + 1e-8)


def infonce_loss(z_c, z_g, kappa):
    z_c = F.normalize(z_c, dim=1)
    z_g = F.normalize(z_g, dim=1)
    sim_matrix = torch.matmul(z_c, z_g.T) / kappa
    labels = torch.arange(z_c.size(0), device=z_c.device)
    return F.cross_entropy(sim_matrix, labels)


def train_sigma(model, optimizer, dataloader, d, lambda_kl, lambda_cox, lambda_cross, kappa):
    model.train()
    total_loss_epoch = 0.0

    for x_c, mask_c, x_g, mask_g, T, E in dataloader:
        x_c = x_c.to(device)
        mask_c = mask_c.to(device)
        x_g = x_g.to(device)
        mask_g = mask_g.to(device)
        T = T.to(device)
        E = E.to(device)

        optimizer.zero_grad()

        x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(x_c, mask_c, x_g, mask_g)

        loss_recon = masked_mse(x_hat_c, x_c, mask_c) + masked_mse(x_hat_g, x_g, mask_g)
        loss_kl = lambda_kl * kl_divergence(mu_shared, log_var_shared)
        loss_cox = lambda_cox * cox_partial_likelihood(eta, T, E)
        loss_cross = lambda_cross * infonce_loss(z_c, z_g, kappa)

        loss = loss_recon + loss_kl + loss_cox + loss_cross

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_epoch += loss.item()

    return total_loss_epoch / len(dataloader)


def evaluate_sigma(model, dataloader):
    model.eval()
    
    all_x_hat_c = []
    all_x_hat_g = []
    all_eta = []
    all_z_shared = []
    all_z_c = []
    all_z_g = []

    with torch.no_grad():
        for x_c, mask_c, x_g, mask_g, T, E in dataloader:
            x_c = x_c.to(device)
            mask_c = mask_c.to(device)
            x_g = x_g.to(device)
            mask_g = mask_g.to(device)
            
            x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, _, _ = model(x_c, mask_c, x_g, mask_g)
            
            all_x_hat_c.append(x_hat_c.cpu())
            all_x_hat_g.append(x_hat_g.cpu())
            all_eta.append(eta.cpu())
            all_z_shared.append(z_shared.cpu())
            all_z_c.append(z_c.cpu())
            all_z_g.append(z_g.cpu())

    return (
        torch.cat(all_x_hat_c, dim=0),
        torch.cat(all_x_hat_g, dim=0),
        torch.cat(all_eta, dim=0),
        torch.cat(all_z_shared, dim=0),
        torch.cat(all_z_c, dim=0),
        torch.cat(all_z_g, dim=0)
    )
