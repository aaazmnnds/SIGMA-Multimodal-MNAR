# Deep learning baseline Optuna tuning and training script for SIGMA study.
import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from lifelines.utils import concordance_index

class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy.core.multiarray':
            module = 'numpy'
        return super().find_class(module, name)

def compute_rmse(imputed, truth, mask):
    obs = mask == 1
    diff = imputed[obs] - truth[obs]
    return np.sqrt(np.mean(diff**2)) if obs.sum() > 0 else 0.0

def compute_mae(imputed, truth, mask):
    obs = mask == 1
    diff = imputed[obs] - truth[obs]
    return np.mean(np.abs(diff)) if obs.sum() > 0 else 0.0


BINARY_COLS_C = [1, 2, 3, 4, 5, 6]
CONTINUOUS_COLS_C = [0, 7, 8]

def masked_recon_loss_c(preds, targets, mask):
    """Mixed MSE + BCE reconstruction loss for clinical features."""
    obs = (1 - mask).bool()
    losses = []
    # MSE for continuous features
    cont_idx = torch.tensor(CONTINUOUS_COLS_C, device=preds.device)
    preds_cont = preds[:, cont_idx]
    targets_cont = targets[:, cont_idx]
    obs_cont = obs[:, cont_idx]
    if obs_cont.sum() > 0:
        mse = ((preds_cont - targets_cont) ** 2 * obs_cont).sum() / (obs_cont.sum() + 1e-8)
        losses.append(mse)
    # BCE for binary features
    bin_idx = torch.tensor(BINARY_COLS_C, device=preds.device)
    preds_bin = preds[:, bin_idx]
    targets_bin = targets[:, bin_idx].clamp(0, 1)
    obs_bin = obs[:, bin_idx]
    if obs_bin.sum() > 0:
        bce = F.binary_cross_entropy_with_logits(
            preds_bin, targets_bin, reduction='none'
        )
        bce_loss = (bce * obs_bin).sum() / (obs_bin.sum() + 1e-8)
        losses.append(bce_loss)
    if not losses:
        return torch.tensor(0.0, device=preds.device)
    return sum(losses) / len(losses)
    # MSE for continuous features
    cont_idx = torch.tensor(CONTINUOUS_COLS_C, device=preds.device)
    preds_cont = preds[:, cont_idx]
    targets_cont = targets[:, cont_idx]
    obs_cont = obs[:, cont_idx]
    if obs_cont.sum() > 0:
        loss += ((preds_cont - targets_cont) ** 2 * obs_cont).sum() / obs_cont.sum()
    # BCE for binary features
    bin_idx = torch.tensor(BINARY_COLS_C, device=preds.device)
    preds_bin = torch.sigmoid(preds[:, bin_idx])
    targets_bin = targets[:, bin_idx]
    obs_bin = obs[:, bin_idx]
    if obs_bin.sum() > 0:
        bce = -(targets_bin * torch.log(preds_bin + 1e-8) + (1 - targets_bin) * torch.log(1 - preds_bin + 1e-8))
        loss += (bce * obs_bin).sum() / obs_bin.sum()
    return loss


class GAIN_Generator(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim*2, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, dim), nn.Sigmoid())
    def forward(self, x, m): return self.fc(torch.cat([x, m], dim=1))

class GAIN_Discriminator(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim*2, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, dim), nn.Sigmoid())
    def forward(self, x, h): return self.fc(torch.cat([x, h], dim=1))

class MIDANet(nn.Module):
    def __init__(self, dim, hidden_dim=256, dropout_rate=0.5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, dim))
    def forward(self, x): return self.decoder(self.encoder(x))

class StandardVAE(nn.Module):
    def __init__(self, p_c, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(p_c, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, d)
        self.fc_var = nn.Linear(256, d)
        
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, p_c)
        )
        self.survival_net = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, 1)
        )
        
    def reparameterize(self, mu, log_var):
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        eta = self.survival_net(z)
        return x_hat, mu, log_var, z, eta

def cox_partial_likelihood(risk, T, E):
    risk = risk.squeeze()
    idx = torch.argsort(T, descending=True)
    risk_sorted = risk[idx]
    E_sorted = E[idx]
    
    hazard_ratio = torch.exp(risk_sorted)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = risk_sorted - log_risk
    loss = -torch.sum(uncensored_likelihood * E_sorted) / (torch.sum(E_sorted) + 1e-8)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['GAIN', 'MIDA', 'StandardVAE'])
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--mechanism', type=str, choices=['threshold', 'selfmask'], default='threshold')
    parser.add_argument('--n_trials', type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BASE = "/home/azman/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    if args.mechanism == 'selfmask':
        RESULTS_DIR = f"{BASE}/results_selfmask"
    else:
        RESULTS_DIR = f"{BASE}/results_final"
    
    out_dir = f"{RESULTS_DIR}/{args.method}_{args.severity}_{args.condition}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[DL-BASELINES-OPTUNA] Method: {args.method} | severity={args.severity} | condition={args.condition} | mechanism={args.mechanism}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    
    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    T = survival['T'].values.astype(np.float32)
    E = survival['E'].values.astype(np.float32)
    
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = NumpyUnpickler(f).load()
        
    from scipy.special import expit
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]

    if args.condition == 'independent':
        if args.mechanism == 'selfmask':
            masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_clinical.npy")
            masks_g = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_genomic.npy")
        else:
            masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
            masks_g = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
        mask_c_features = masks_c[severity_idx].astype(np.float32)
        mask_g_expr = masks_g[severity_idx].astype(np.float32)
    else:  # joint
        if args.mechanism == 'selfmask':
            calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results_selfmask.csv")
        else:
            calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
        alpha_c, beta_c = get_params('clinical', args.severity)
        alpha_g, beta_g = get_params('genomic', args.severity)
        np.random.seed(42)
        X_c_features = X_c[:, 1:]
        X_expr = X_g[:, :49]
        if args.mechanism == 'selfmask':
            prob_c = expit(alpha_c * np.abs(X_c_features) + beta_c)
            prob_g = expit(alpha_g * np.abs(X_expr) + beta_g)
        else:
            age = X_c[:, 0]
            tau = np.median(age)
            age_indicator = (age > tau).astype(np.float32)
            prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features) + beta_c)
            prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        mask_g_expr = np.random.binomial(1, prob_g).astype(np.float32)

    age_mask_col = np.zeros((X_c.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    mask_g_full = np.zeros((X_g.shape[0], X_g.shape[1]), dtype=np.float32)
    mask_g_full[:, :49] = mask_g_expr
    
    mask_g_expr_full = mask_g_expr  # alias just in case
    
    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr == 1] = 0.0
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    print(f"[DL-BASELINES-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    
    def objective(trial):
        if args.method == 'GAIN':
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            alpha = trial.suggest_float('alpha', 10.0, 200.0)
            hint_rate = trial.suggest_float('hint_rate', 0.5, 0.95)
            hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
            iters = trial.suggest_int('iters', 1000, 5000)
            
            dim_c = X_c.shape[1]
            G_c = GAIN_Generator(dim_c, hidden_dim).to(device)
            D_c = GAIN_Discriminator(dim_c, hidden_dim).to(device)
            G_c_opt = torch.optim.Adam(G_c.parameters(), lr=lr)
            D_c_opt = torch.optim.Adam(D_c.parameters(), lr=lr)
            
            dim_g = X_g.shape[1]
            G_g = GAIN_Generator(dim_g, hidden_dim).to(device)
            D_g = GAIN_Discriminator(dim_g, hidden_dim).to(device)
            G_g_opt = torch.optim.Adam(G_g.parameters(), lr=lr)
            D_g_opt = torch.optim.Adam(D_g.parameters(), lr=lr)
            
            def train_gain(X_tr, M_tr, G, D, G_opt, D_opt, dim):
                n_samples = X_tr.shape[0]
                for _ in range(iters):
                    idx = np.random.choice(n_samples, 128)
                    X_mb = torch.FloatTensor(X_tr[idx]).to(device)
                    M_obs = 1.0 - torch.FloatTensor(M_tr[idx]).to(device)
                    Z_mb = torch.rand(128, dim).to(device)
                    hint_mask = (torch.rand(128, dim).to(device) < hint_rate).float()
                    H_mb = hint_mask * M_obs + (1 - hint_mask) * 0.5
                    X_corrupted = M_obs * X_mb + (1 - M_obs) * Z_mb
                    
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
                    D_prob = D(X_hat.detach(), H_mb)
                    D_loss = -torch.mean(M_obs * torch.log(D_prob + 1e-8) + (1 - M_obs) * torch.log(1 - D_prob + 1e-8))
                    
                    D_opt.zero_grad()
                    D_loss.backward()
                    D_opt.step()
                    
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
                    D_prob = D(X_hat, H_mb)
                    G_loss_temp = -torch.mean((1 - M_obs) * torch.log(D_prob + 1e-8))
                    MSE_loss = torch.mean((M_obs * G_sample - M_obs * X_mb)**2) / (torch.mean(M_obs) + 1e-8)
                    G_loss = G_loss_temp + alpha * MSE_loss
                    
                    G_opt.zero_grad()
                    G_loss.backward()
                    G_opt.step()
                    
            train_gain(X_c_masked[train_idx_0], mask_c_full[train_idx_0], G_c, D_c, G_c_opt, D_c_opt, dim_c)
            train_gain(X_g_masked[train_idx_0], mask_g_full[train_idx_0], G_g, D_g, G_g_opt, D_g_opt, dim_g)
            
            def test_gain(G, X_te, M_te, dim):
                G.eval()
                with torch.no_grad():
                    X_ts = torch.FloatTensor(X_te).to(device)
                    M_obs = 1.0 - torch.FloatTensor(M_te).to(device)
                    Z_ts = torch.rand(X_ts.shape[0], dim).to(device)
                    X_corrupted = M_obs * X_ts + (1 - M_obs) * Z_ts
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_ts + (1 - M_obs) * G_sample
                return X_hat.cpu().numpy()
                
            X_c_imp = test_gain(G_c, X_c_masked[val_idx_0], mask_c_full[val_idx_0], dim_c)
            X_g_imp = test_gain(G_g, X_g_masked[val_idx_0], mask_g_full[val_idx_0], dim_g)
            
            rmse_c = compute_rmse(X_c_imp, X_c[val_idx_0], mask_c_full[val_idx_0])
            rmse_g = compute_rmse(X_g_imp[:, :49], X_g[val_idx_0][:, :49], mask_g_expr[val_idx_0])
            val_obj = rmse_c + rmse_g
            
        elif args.method == 'MIDA':
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            latent_dim_offset = trial.suggest_int('latent_dim_offset', 5, 20)
            epochs = trial.suggest_int('epochs', 50, 150)
            
            dim_c = X_c.shape[1]
            dim_g = X_g.shape[1]
            hidden_dim_c = dim_c + latent_dim_offset
            hidden_dim_g = dim_g + latent_dim_offset
            model_c = MIDANet(dim_c, hidden_dim_c, dropout_rate=0.5).to(device)
            model_g = MIDANet(dim_g, hidden_dim_g, dropout_rate=0.5).to(device)
            opt_c = torch.optim.Adam(model_c.parameters(), lr=lr)
            opt_g = torch.optim.Adam(model_g.parameters(), lr=lr)
            
            def train_mida(model, opt, X_tr, M_tr):
                dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(M_tr))
                loader = DataLoader(dataset, batch_size=128, shuffle=True)
                for _ in range(epochs):
                    model.train()
                    for x_mb, m_mb in loader:
                        x_mb = x_mb.to(device)
                        M_obs = 1.0 - m_mb.to(device)
                        drop_mask = (torch.rand(x_mb.shape).to(device) > 0.5).float()
                        M_corrupted = M_obs * drop_mask
                        X_corrupted = x_mb * M_corrupted
                        x_hat = model(X_corrupted)
                        loss = torch.sum(((x_hat - x_mb) * M_obs)**2) / (torch.sum(M_obs) + 1e-8)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        
            train_mida(model_c, opt_c, X_c_masked[train_idx_0], mask_c_full[train_idx_0])
            train_mida(model_g, opt_g, X_g_masked[train_idx_0], mask_g_full[train_idx_0])
            
            def test_mida(model, X_te, M_te):
                model.eval()
                X_ts = torch.FloatTensor(X_te).to(device)
                M_obs = 1.0 - torch.FloatTensor(M_te).to(device)
                preds = []
                with torch.no_grad():
                    for _ in range(5):
                        drop_mask = (torch.rand(X_ts.shape).to(device) > 0.5).float()
                        M_corrupted = M_obs * drop_mask
                        X_corrupted = X_ts * M_corrupted
                        preds.append(model(X_corrupted))
                    mean_pred = torch.stack(preds).mean(dim=0)
                    X_hat = M_obs * X_ts + (1 - M_obs) * mean_pred
                return X_hat.cpu().numpy()
                
            X_c_imp = test_mida(model_c, X_c_masked[val_idx_0], mask_c_full[val_idx_0])
            X_g_imp = test_mida(model_g, X_g_masked[val_idx_0], mask_g_full[val_idx_0])
            
            rmse_c = compute_rmse(X_c_imp, X_c[val_idx_0], mask_c_full[val_idx_0])
            rmse_g = compute_rmse(X_g_imp[:, :49], X_g[val_idx_0][:, :49], mask_g_expr[val_idx_0])
            val_obj = rmse_c + rmse_g
            
        elif args.method == 'StandardVAE':
            d = trial.suggest_categorical('d', [32, 64, 128])
            lambda_kl = trial.suggest_float('lambda_kl', 1e-2, 2.0, log=True)
            lambda_cox = trial.suggest_float('lambda_cox', 0.1, 5.0, log=True)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            p_c = X_c.shape[1]
            model = StandardVAE(p_c, d).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_c_masked[train_idx_0]),
                torch.FloatTensor(mask_c_full[train_idx_0]),
                torch.FloatTensor(T[train_idx_0]),
                torch.FloatTensor(E[train_idx_0])
            )
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            for epoch in range(50):
                model.train()
                for x_mb, m_mb, t_mb, e_mb in train_loader:
                    x_mb = x_mb.to(device)
                    m_mb = m_mb.to(device)
                    t_mb = t_mb.to(device)
                    e_mb = e_mb.to(device)
                    
                    x_hat, mu, log_var, z, eta = model(x_mb)
                    
                    M_obs = 1.0 - m_mb
                    recon_loss = masked_recon_loss_c(x_hat, x_mb, m_mb)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_mb.shape[0]
                    cox_loss = cox_partial_likelihood(eta, t_mb, e_mb)
                    
                    loss = recon_loss + lambda_kl * kl_loss + lambda_cox * cox_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                x_val = torch.FloatTensor(X_c_masked[val_idx_0]).to(device)
                m_val = torch.FloatTensor(mask_c_full[val_idx_0]).to(device)
                t_val = torch.FloatTensor(T[val_idx_0]).to(device)
                e_val = torch.FloatTensor(E[val_idx_0]).to(device)
                
                x_hat, _, _, _, eta = model(x_val)
                X_c_imp = x_hat.cpu().numpy()
                if args.method == 'StandardVAE':
                    bin_cols = [1, 2, 3, 4, 5, 6]
                    X_c_imp[:, bin_cols] = 1 / (1 + np.exp(-X_c_imp[:, bin_cols]))
                eta_ts = eta.cpu().numpy()
                
                val_recon_loss = compute_rmse(X_c_imp, X_c[val_idx_0], mask_c_full[val_idx_0])
                try:
                    from sksurv.metrics import concordance_index_censored
                    eta_val = np.nan_to_num(eta_ts.flatten(), nan=0.0, posinf=1e6, neginf=-1e6)
                    val_c_index = concordance_index_censored(E[val_idx_0].astype(bool), T[val_idx_0], eta_val)[0]
                except:
                    val_c_index = 0.5
                    
                val_obj = val_recon_loss - 2.0 * val_c_index
        
        return val_obj

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    best_loss = study.best_value
    
    params_str = " | ".join([f"{k}={v}" for k, v in best_params.items()])
    print(f"[DL-BASELINES-OPTUNA] Best params: {params_str} | val_obj={best_loss:.4f}")
    
    with open(f"{out_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("[DL-BASELINES-OPTUNA] Retraining on all 5 folds...")
    
    results_list = []
    
    for k, fold_dict in enumerate(folds, 1):
        train_idx = fold_dict['train']
        test_idx = fold_dict['test']
        
        if args.method == 'GAIN':
            dim_c = X_c.shape[1]
            G_c = GAIN_Generator(dim_c, best_params['hidden_dim']).to(device)
            D_c = GAIN_Discriminator(dim_c, best_params['hidden_dim']).to(device)
            G_c_opt = torch.optim.Adam(G_c.parameters(), lr=best_params['lr'])
            D_c_opt = torch.optim.Adam(D_c.parameters(), lr=best_params['lr'])
            
            dim_g = X_g.shape[1]
            G_g = GAIN_Generator(dim_g, best_params['hidden_dim']).to(device)
            D_g = GAIN_Discriminator(dim_g, best_params['hidden_dim']).to(device)
            G_g_opt = torch.optim.Adam(G_g.parameters(), lr=best_params['lr'])
            D_g_opt = torch.optim.Adam(D_g.parameters(), lr=best_params['lr'])
            
            def train_gain(X_tr, M_tr, G, D, G_opt, D_opt, dim):
                n_samples = X_tr.shape[0]
                for _ in range(best_params['iters']):
                    idx = np.random.choice(n_samples, 128)
                    X_mb = torch.FloatTensor(X_tr[idx]).to(device)
                    M_obs = 1.0 - torch.FloatTensor(M_tr[idx]).to(device)
                    Z_mb = torch.rand(128, dim).to(device)
                    hint_mask = (torch.rand(128, dim).to(device) < best_params['hint_rate']).float()
                    H_mb = hint_mask * M_obs + (1 - hint_mask) * 0.5
                    X_corrupted = M_obs * X_mb + (1 - M_obs) * Z_mb
                    
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
                    D_prob = D(X_hat.detach(), H_mb)
                    D_loss = -torch.mean(M_obs * torch.log(D_prob + 1e-8) + (1 - M_obs) * torch.log(1 - D_prob + 1e-8))
                    
                    D_opt.zero_grad()
                    D_loss.backward()
                    D_opt.step()
                    
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
                    D_prob = D(X_hat, H_mb)
                    G_loss_temp = -torch.mean((1 - M_obs) * torch.log(D_prob + 1e-8))
                    MSE_loss = torch.mean((M_obs * G_sample - M_obs * X_mb)**2) / (torch.mean(M_obs) + 1e-8)
                    G_loss = G_loss_temp + best_params['alpha'] * MSE_loss
                    
                    G_opt.zero_grad()
                    G_loss.backward()
                    G_opt.step()
                    
            train_gain(X_c_masked[train_idx], mask_c_full[train_idx], G_c, D_c, G_c_opt, D_c_opt, dim_c)
            train_gain(X_g_masked[train_idx], mask_g_full[train_idx], G_g, D_g, G_g_opt, D_g_opt, dim_g)
            
            def test_gain(G, X_te, M_te, dim):
                G.eval()
                with torch.no_grad():
                    X_ts = torch.FloatTensor(X_te).to(device)
                    M_obs = 1.0 - torch.FloatTensor(M_te).to(device)
                    Z_ts = torch.rand(X_ts.shape[0], dim).to(device)
                    X_corrupted = M_obs * X_ts + (1 - M_obs) * Z_ts
                    G_sample = G(X_corrupted, M_obs)
                    X_hat = M_obs * X_ts + (1 - M_obs) * G_sample
                return X_hat.cpu().numpy()
                
            X_c_imp = test_gain(G_c, X_c_masked[test_idx], mask_c_full[test_idx], dim_c)
            X_g_imp = test_gain(G_g, X_g_masked[test_idx], mask_g_full[test_idx], dim_g)
            
        elif args.method == 'MIDA':
            dim_c = X_c.shape[1]
            dim_g = X_g.shape[1]
            hidden_dim_c = dim_c + best_params['latent_dim_offset']
            hidden_dim_g = dim_g + best_params['latent_dim_offset']
            model_c = MIDANet(dim_c, hidden_dim_c, dropout_rate=0.5).to(device)
            model_g = MIDANet(dim_g, hidden_dim_g, dropout_rate=0.5).to(device)
            opt_c = torch.optim.Adam(model_c.parameters(), lr=best_params['lr'])
            opt_g = torch.optim.Adam(model_g.parameters(), lr=best_params['lr'])
            
            def train_mida(model, opt, X_tr, M_tr):
                dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(M_tr))
                loader = DataLoader(dataset, batch_size=128, shuffle=True)
                for _ in range(best_params['epochs']):
                    model.train()
                    for x_mb, m_mb in loader:
                        x_mb = x_mb.to(device)
                        M_obs = 1.0 - m_mb.to(device)
                        drop_mask = (torch.rand(x_mb.shape).to(device) > 0.5).float()
                        M_corrupted = M_obs * drop_mask
                        X_corrupted = x_mb * M_corrupted
                        x_hat = model(X_corrupted)
                        loss = torch.sum(((x_hat - x_mb) * M_obs)**2) / (torch.sum(M_obs) + 1e-8)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        
            train_mida(model_c, opt_c, X_c_masked[train_idx], mask_c_full[train_idx])
            train_mida(model_g, opt_g, X_g_masked[train_idx], mask_g_full[train_idx])
            
            def test_mida(model, X_te, M_te):
                model.eval()
                X_ts = torch.FloatTensor(X_te).to(device)
                M_obs = 1.0 - torch.FloatTensor(M_te).to(device)
                preds = []
                with torch.no_grad():
                    for _ in range(5):
                        drop_mask = (torch.rand(X_ts.shape).to(device) > 0.5).float()
                        M_corrupted = M_obs * drop_mask
                        X_corrupted = X_ts * M_corrupted
                        preds.append(model(X_corrupted))
                    mean_pred = torch.stack(preds).mean(dim=0)
                    X_hat = M_obs * X_ts + (1 - M_obs) * mean_pred
                return X_hat.cpu().numpy()
                
            X_c_imp = test_mida(model_c, X_c_masked[test_idx], mask_c_full[test_idx])
            X_g_imp = test_mida(model_g, X_g_masked[test_idx], mask_g_full[test_idx])
            
        elif args.method == 'StandardVAE':
            p_c = X_c.shape[1]
            model = StandardVAE(p_c, best_params['d']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_c_masked[train_idx]),
                torch.FloatTensor(mask_c_full[train_idx]),
                torch.FloatTensor(T[train_idx]),
                torch.FloatTensor(E[train_idx])
            )
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            for epoch in range(50):
                model.train()
                for x_mb, m_mb, t_mb, e_mb in train_loader:
                    x_mb = x_mb.to(device)
                    m_mb = m_mb.to(device)
                    t_mb = t_mb.to(device)
                    e_mb = e_mb.to(device)
                    
                    x_hat, mu, log_var, z, eta = model(x_mb)
                    
                    M_obs = 1.0 - m_mb
                    recon_loss = masked_recon_loss_c(x_hat, x_mb, m_mb)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_mb.shape[0]
                    cox_loss = cox_partial_likelihood(eta, t_mb, e_mb)
                    
                    loss = recon_loss + best_params['lambda_kl'] * kl_loss + best_params['lambda_cox'] * cox_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                x_test = torch.FloatTensor(X_c_masked[test_idx]).to(device)
                m_test = torch.FloatTensor(mask_c_full[test_idx]).to(device)
                X_c_imp, _, _, _, eta_ts = model(x_test)
                if args.method == 'StandardVAE':
                    bin_cols = [1, 2, 3, 4, 5, 6]
                    X_c_imp[:, bin_cols] = torch.sigmoid(X_c_imp[:, bin_cols])
                M_obs = 1.0 - m_test
                X_c_imp = M_obs * x_test + (1 - M_obs) * X_c_imp
                X_c_imp = X_c_imp.cpu().numpy()
                X_g_imp = X_g_masked[test_idx].copy()
                eta_ts = eta_ts.cpu().numpy()
        
        rmse_c = compute_rmse(X_c_imp, X_c[test_idx], mask_c_full[test_idx])
        mae_c = compute_mae(X_c_imp, X_c[test_idx], mask_c_full[test_idx])
        
        if args.method == 'StandardVAE':
            rmse_g = 0.0
            mae_g = 0.0
        else:
            rmse_g = compute_rmse(X_g_imp[:, :49], X_g[test_idx][:, :49], mask_g_expr[test_idx])
            mae_g = compute_mae(X_g_imp[:, :49], X_g[test_idx][:, :49], mask_g_expr[test_idx])
            
        print(f"[DL-BASELINES-OPTUNA] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f} | RMSE_g: {rmse_g:.4f} | MAE_g: {mae_g:.4f}")
        
        results_list.append({'Fold': k, 'RMSE_c': rmse_c, 'MAE_c': mae_c, 'RMSE_g': rmse_g, 'MAE_g': mae_g})
        
        fold_res = {
            'test_idx': test_idx,
            'x_hat_c': X_c_imp,
            'x_hat_g': X_g_imp,
            'rmse_c': rmse_c,
            'rmse_g': rmse_g,
            'mae_c': mae_c,
            'mae_g': mae_g
        }
        if args.method == 'StandardVAE':
            fold_res['eta'] = eta_ts
            
        with open(f"{out_dir}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump(fold_res, f)
            
    res_df = pd.DataFrame(results_list)
    mean_rmse_c = res_df['RMSE_c'].mean()
    std_rmse_c = res_df['RMSE_c'].std()
    mean_rmse_g = res_df['RMSE_g'].mean()
    std_rmse_g = res_df['RMSE_g'].std()
    
    print(f"[DL-BASELINES-OPTUNA] Summary | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | RMSE_g: {mean_rmse_g:.4f} ± {std_rmse_g:.4f}")
    res_df.to_csv(f"{out_dir}/summary.csv", index=False)

if __name__ == '__main__':
    main()
