# MNARSVAE Optuna tuning script for SIGMA study. Unimodal clinical-only baseline.
import os, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def masked_mse(preds, targets, mask):
    diff = preds - targets
    diff = diff * (1 - mask)
    return (diff ** 2).sum() / max(1.0, (1 - mask).sum().item())

def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

def cox_partial_likelihood(eta, t, e):
    idx = torch.argsort(t, descending=True)
    eta = eta[idx]
    e = e[idx]
    risk_scores = torch.exp(eta)
    risk_sums = torch.cumsum(risk_scores, dim=0)
    log_risk_sums = torch.log(risk_sums + 1e-8)
    loss = -torch.sum((eta - log_risk_sums) * e) / max(1.0, e.sum().item())
    return loss


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


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, out_dim)
        self.logvar = nn.Linear(64, out_dim)
    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.fc(x)

class SurvivalNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x)

class MNARSVAEModel(nn.Module):
    def __init__(self, p_c, d):
        super().__init__()
        self.encoder = Encoder(p_c * 2, d)
        self.decoder = Decoder(d, p_c)
        self.survival_net = SurvivalNet(d)
        
    def forward(self, x_c, mask_c):
        x_in = torch.cat([x_c, mask_c], dim=1)
        mu, log_var = self.encoder(x_in)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        x_hat_c = self.decoder(z)
        eta = self.survival_net(z)
        return x_hat_c, mu, log_var, z, eta

def objective(trial, train_loader, val_loader, p_c, batch_size, tune_epochs, device):
    d = trial.suggest_categorical('d', [32, 64, 128])
    lambda_kl = trial.suggest_float('lambda_kl', 1e-2, 2.0, log=True)
    lambda_cox = trial.suggest_float('lambda_cox', 0.1, 5.0, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    model = MNARSVAEModel(p_c, d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val = float('inf')
    patience_count = 0
    
    for epoch in range(tune_epochs):
        model.train()
        for xc_mb, mc_mb, t_mb, e_mb in train_loader:
            xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
            t_mb, e_mb = t_mb.to(device), e_mb.to(device)
            
            optimizer.zero_grad()
            x_hat_c, mu, log_var, z, eta = model(xc_mb, mc_mb)
            
            l_recon = masked_recon_loss_c(x_hat_c, xc_mb, mc_mb)
            l_kl = kl_divergence(mu, log_var)
            l_cox = cox_partial_likelihood(eta, t_mb, e_mb)
            
            loss = l_recon + lambda_kl * l_kl + lambda_cox * l_cox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        val_recon_loss = 0.0
        all_eta_val = []
        all_T_val = []
        all_E_val = []
        with torch.no_grad():
            for xc_mb, mc_mb, t_mb, e_mb in val_loader:
                xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                
                x_hat_c, _, _, _, eta = model(xc_mb, mc_mb)
                val_recon_loss += masked_recon_loss_c(x_hat_c, xc_mb, mc_mb).item()
                
                all_eta_val.append(eta.cpu().numpy().flatten())
                all_T_val.append(t_mb.cpu().numpy())
                all_E_val.append(e_mb.cpu().numpy())
                
        val_recon_loss /= len(val_loader)
        
        try:
            from sksurv.metrics import concordance_index_censored
            eta_val = np.concatenate(all_eta_val)
            T_val = np.concatenate(all_T_val)
            E_val = np.concatenate(all_E_val)
            eta_val = np.nan_to_num(eta_val, nan=0.0, posinf=1e6, neginf=-1e6)
            val_c_index = concordance_index_censored(E_val.astype(bool), T_val, eta_val)[0]
        except:
            val_c_index = 0.5
            
        val_loss = val_recon_loss - 2.0 * val_c_index
        
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
        else:
            patience_count += 1
            
        if patience_count >= 10:
            break
            
    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light','moderate','severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent','joint'])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--tune_epochs', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    BASE = "/home/azman/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    
    out_dir = f"{RESULTS_DIR}/{args.severity}_{args.condition}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[MNARSVAE-OPTUNA] Config: severity={args.severity} | condition={args.condition}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")

    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)

    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    T = survival['T'].values.astype(np.float32)
    E = survival['E'].values.astype(np.float32)
    p_c = X_c.shape[1]

    from scipy.special import expit
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]

    if args.condition == 'independent':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        mask_c_features = masks_c[severity_idx].astype(np.float32)
    else:  # joint
        age = X_c[:, 0]
        tau = np.median(age)
        age_indicator = (age > tau).astype(np.float32)
        calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
        alpha_c, beta_c = get_params('clinical', args.severity)
        np.random.seed(42)
        X_c_features = X_c[:, 1:]
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)

    age_mask_col = np.zeros((X_c.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)

    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    

    print(f"[MNARSVAE-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    def make_loader_0(idx, shuffle):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X_c_masked[idx]),
            torch.FloatTensor(mask_c_full[idx]),
            torch.FloatTensor(T[idx]),
            torch.FloatTensor(E[idx])
        ), batch_size=args.batch_size, shuffle=shuffle)

    train_loader_0 = make_loader_0(train_idx_0, True)
    val_loader_0 = make_loader_0(val_idx_0, False)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )
    study.optimize(lambda trial: objective(trial, train_loader_0, val_loader_0, p_c, args.batch_size, args.tune_epochs, device), n_trials=args.n_trials)
    
    best_params = study.best_params
    best_loss = study.best_value
    print(f"[MNARSVAE-OPTUNA] Fold 0 Best trial: val_loss={best_loss:.4f} | d={best_params['d']} | lambda_kl={best_params['lambda_kl']} | lambda_cox={best_params['lambda_cox']} | lr={best_params['lr']}")
    
    with open(f"{out_dir}/best_params.json", 'w') as f:
        import json
        json.dump(best_params, f, indent=4)
        
    print(f"[MNARSVAE-OPTUNA] Retraining on all 5 folds with best params...")

    all_rmse_c = []
    all_mae_c = []

    for k, fold_dict in enumerate(folds, 1):
        train_idx = fold_dict['train']
        val_idx = fold_dict['val']
        test_idx = fold_dict['test']
        
        def make_loader(idx, shuffle):
            from torch.utils.data import TensorDataset, DataLoader
            return DataLoader(TensorDataset(
                torch.FloatTensor(X_c_masked[idx]),
                torch.FloatTensor(mask_c_full[idx]),
                torch.FloatTensor(T[idx]),
                torch.FloatTensor(E[idx])
            ), batch_size=args.batch_size, shuffle=shuffle)

        train_loader = make_loader(train_idx, True)
        val_loader = make_loader(val_idx, False)
        test_loader = make_loader(test_idx, False)
        
        model = MNARSVAEModel(p_c, best_params['d']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        
        best_val_retrain = float('inf')
        patience_counter = 0
        best_state = None
        
        for ep in range(args.max_epochs):
            model.train()
            for xc_mb, mc_mb, t_mb, e_mb in train_loader:
                xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                
                optimizer.zero_grad()
                x_hat_c, mu, log_var, z, eta = model(xc_mb, mc_mb)
                l_recon = masked_recon_loss_c(x_hat_c, xc_mb, mc_mb)
                l_kl = kl_divergence(mu, log_var)
                l_cox = cox_partial_likelihood(eta, t_mb, e_mb)
                
                loss = l_recon + best_params['lambda_kl'] * l_kl + best_params['lambda_cox'] * l_cox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            model.eval()
            val_recon_loss = 0.0
            all_eta_val = []
            all_T_val = []
            all_E_val = []
            with torch.no_grad():
                for xc_mb, mc_mb, t_mb, e_mb in val_loader:
                    xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                    t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                    x_hat_c, _, _, _, eta = model(xc_mb, mc_mb)
                    val_recon_loss += masked_recon_loss_c(x_hat_c, xc_mb, mc_mb).item()
                    all_eta_val.append(eta.cpu().numpy().flatten())
                    all_T_val.append(t_mb.cpu().numpy())
                    all_E_val.append(e_mb.cpu().numpy())
                    
            val_recon_loss /= len(val_loader)
            
            try:
                from sksurv.metrics import concordance_index_censored
                eta_val = np.concatenate(all_eta_val)
                T_val = np.concatenate(all_T_val)
                E_val = np.concatenate(all_E_val)
                eta_val = np.nan_to_num(eta_val, nan=0.0, posinf=1e6, neginf=-1e6)
                val_c_index = concordance_index_censored(E_val.astype(bool), T_val, eta_val)[0]
            except:
                val_c_index = 0.5
                
            val_loss = val_recon_loss - 2.0 * val_c_index
            
            if val_loss < best_val_retrain:
                best_val_retrain = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                break
                
        model.load_state_dict(best_state)
        model.eval()
        
        x_hat_c_full = []
        eta_full = []
        with torch.no_grad():
            for xc_mb, mc_mb, t_mb, e_mb in test_loader:
                xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                x_hat_c, _, _, _, eta = model(xc_mb, mc_mb)
                x_hat_c_full.append(x_hat_c.cpu().numpy())
                eta_full.append(eta.cpu().numpy())
                
        x_hat_c_np = np.concatenate(x_hat_c_full)
        bin_cols = [1, 2, 3, 4, 5, 6]
        x_hat_c_np[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c_np[:, bin_cols]))
        eta_np = np.concatenate(eta_full)
        
        mask_test_c = mask_c_full[test_idx]
        x_true_c = X_c[test_idx]
        obs_c = mask_test_c == 1
        diff_c = x_hat_c_np[obs_c] - x_true_c[obs_c]
        rmse_c = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
        mae_c = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
        
        all_rmse_c.append(rmse_c)
        all_mae_c.append(mae_c)
        
        print(f"[MNARSVAE-OPTUNA] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f}")
        
        res = {
            'test_idx': test_idx,
            'x_hat_c': x_hat_c_np,
            'eta': eta_np,
            'rmse_c': rmse_c,
            'mae_c': mae_c,
            'best_params': best_params
        }
        with open(f"{out_dir}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump(res, f)
            
    summary = {
        'Fold': list(range(1, 6)) + ['Mean', 'Std'],
        'RMSE_c': all_rmse_c + [np.mean(all_rmse_c), np.std(all_rmse_c)],
        'MAE_c': all_mae_c + [np.mean(all_mae_c), np.std(all_mae_c)]
    }
    pd.DataFrame(summary).to_csv(f"{out_dir}/summary.csv", index=False)

if __name__ == '__main__':
    main()
