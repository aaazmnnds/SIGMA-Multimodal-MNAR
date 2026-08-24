# MyeVAE Optuna hyperparameter tuning implementation for SIGMA study. Based on Chang et al. 2025.
import os, pickle, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit
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


class MyeVAE(nn.Module):
    def __init__(self, p_c, p_g, d):
        super().__init__()
        self.enc_c_fc = nn.Sequential(nn.Linear(2*p_c, 64), nn.ReLU())
        self.enc_c_mu = nn.Linear(64, d)
        self.enc_c_logvar = nn.Linear(64, d)
        
        self.enc_g_fc = nn.Sequential(nn.Linear(2*p_g, 64), nn.ReLU())
        self.enc_g_mu = nn.Linear(64, d)
        self.enc_g_logvar = nn.Linear(64, d)
        
        self.dec_c = nn.Sequential(nn.Linear(2*d, 64), nn.ReLU(), nn.Linear(64, p_c))
        self.dec_g = nn.Sequential(nn.Linear(2*d, 64), nn.ReLU(), nn.Linear(64, p_g))
        
        self.surv = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, 1))
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def forward(self, x_c, mask_c, x_g, mask_g):
        in_c = torch.cat([x_c, mask_c], dim=1)
        h_c = self.enc_c_fc(in_c)
        mu_c = self.enc_c_mu(h_c)
        log_var_c = self.enc_c_logvar(h_c)
        
        in_g = torch.cat([x_g, mask_g], dim=1)
        h_g = self.enc_g_fc(in_g)
        mu_g = self.enc_g_mu(h_g)
        log_var_g = self.enc_g_logvar(h_g)
        
        z_c = self.reparameterize(mu_c, log_var_c)
        z_g = self.reparameterize(mu_g, log_var_g)
        
        z_shared = torch.cat([z_c, z_g], dim=1)
        
        x_hat_c = self.dec_c(z_shared)
        x_hat_g = self.dec_g(z_shared)
        
        eta = self.surv(z_shared)
        return x_hat_c, x_hat_g, eta, mu_c, log_var_c, mu_g, log_var_g, z_shared

def objective(trial, train_loader, val_loader, p_c, p_g, batch_size, tune_epochs, device):
    d = trial.suggest_categorical('d', [32, 64, 128])
    lambda_kl = trial.suggest_float('lambda_kl', 1e-2, 2.0, log=True)
    lambda_cox = trial.suggest_float('lambda_cox', 0.1, 5.0, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    model = MyeVAE(p_c, p_g, d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val = float('inf')
    patience_count = 0
    
    for epoch in range(tune_epochs):
        model.train()
        for xc_mb, mc_mb, xg_mb, mg_mb, t_mb, e_mb in train_loader:
            xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
            xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
            t_mb, e_mb = t_mb.to(device), e_mb.to(device)
            
            optimizer.zero_grad()
            x_hat_c, x_hat_g, eta, mu_c, log_var_c, mu_g, log_var_g, z_shared = model(xc_mb, mc_mb, xg_mb, mg_mb)
            
            l_recon = masked_recon_loss_c(x_hat_c, xc_mb, mc_mb) + masked_mse(x_hat_g, xg_mb, mg_mb)
            l_kl = kl_divergence(mu_c, log_var_c) + kl_divergence(mu_g, log_var_g)
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
            for xc_mb, mc_mb, xg_mb, mg_mb, t_mb, e_mb in val_loader:
                xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
                t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                
                x_hat_c, x_hat_g, eta, _, _, _, _, _ = model(xc_mb, mc_mb, xg_mb, mg_mb)
                l_c = masked_recon_loss_c(x_hat_c, xc_mb, mc_mb)
                l_g = masked_mse(x_hat_g, xg_mb, mg_mb)
                val_recon_loss += (l_c + l_g).item()
                
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
            
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--tune_epochs', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    BASE = "/home/azman/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    FOLDS_DIR = f"{PROCESSED_DIR}/folds"
    RESULTS_DIR = f"{BASE}/results_final"
    
    out_path = f"{RESULTS_DIR}/{args.severity}_{args.condition}"
    os.makedirs(out_path, exist_ok=True)
    print(f"[MYEVAE-OPTUNA] Config: severity={args.severity} | condition={args.condition}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    
    with open(f"{FOLDS_DIR}/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)
        
    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    genomic_cols = genomic.drop(columns=['PATIENT_ID']).columns.tolist()
    expr_cols = [c for c in genomic_cols if not c.endswith('_CNA')]
    X_expr = genomic[expr_cols].values.astype(np.float32)
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    
    T = survival['T'].values.astype(np.float32)
    E = survival['E'].values.astype(np.float32)
    
    masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
    masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
    
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]
    mask_c_features = masks_c[severity_idx]
    age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    mask_g_expr_full = masks_g_expr[severity_idx]
    
    if args.condition == 'joint':
        age = X_c[:, 0]
        tau = np.median(age)
        age_indicator = (age > tau).astype(np.float32)
        calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
        alpha_c, beta_c = get_params('clinical', args.severity)
        alpha_g, beta_g = get_params('genomic', args.severity)
        
        np.random.seed(42)
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c[:, 1:]) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        mask_c_full = np.concatenate([np.zeros((X_c.shape[0], 1), dtype=np.float32), mask_c_features], axis=1)
        prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)
        mask_g_expr_full = np.random.binomial(1, prob_g).astype(np.float32)
        
    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = np.nan
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = np.nan
    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full
    
    p_c = X_c.shape[1]
    p_g = X_g.shape[1]
    
    results = []

    print(f"[MYEVAE-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    Xc_tr_0, Mc_tr_0 = X_c_masked[train_idx_0].copy(), mask_c_full[train_idx_0]
    Xg_tr_0, Mg_tr_0 = X_g_masked[train_idx_0].copy(), mask_g_full[train_idx_0]
    Xc_vl_0, Mc_vl_0 = X_c_masked[val_idx_0].copy(), mask_c_full[val_idx_0]
    Xg_vl_0, Mg_vl_0 = X_g_masked[val_idx_0].copy(), mask_g_full[val_idx_0]
    
    c_means_0 = np.nanmean(Xc_tr_0, axis=0)
    g_means_0 = np.nanmean(Xg_tr_0, axis=0)
    c_means_0[np.isnan(c_means_0)] = 0.0
    g_means_0[np.isnan(g_means_0)] = 0.0
    
    for i in range(p_c):
        Xc_tr_0[np.isnan(Xc_tr_0[:, i]), i] = c_means_0[i]
        Xc_vl_0[np.isnan(Xc_vl_0[:, i]), i] = c_means_0[i]
    for i in range(p_g):
        Xg_tr_0[np.isnan(Xg_tr_0[:, i]), i] = g_means_0[i]
        Xg_vl_0[np.isnan(Xg_vl_0[:, i]), i] = g_means_0[i]
        
    ds_train_0 = TensorDataset(torch.FloatTensor(Xc_tr_0), torch.FloatTensor(Mc_tr_0), torch.FloatTensor(Xg_tr_0), torch.FloatTensor(Mg_tr_0), torch.FloatTensor(T[train_idx_0]), torch.FloatTensor(E[train_idx_0]))
    ds_val_0 = TensorDataset(torch.FloatTensor(Xc_vl_0), torch.FloatTensor(Mc_vl_0), torch.FloatTensor(Xg_vl_0), torch.FloatTensor(Mg_vl_0), torch.FloatTensor(T[val_idx_0]), torch.FloatTensor(E[val_idx_0]))
    dl_train_0 = DataLoader(ds_train_0, batch_size=args.batch_size, shuffle=True)
    dl_val_0 = DataLoader(ds_val_0, batch_size=args.batch_size, shuffle=False)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )
    study.optimize(
        lambda trial: objective(trial, dl_train_0, dl_val_0, p_c, p_g, args.batch_size, args.tune_epochs, device),
        n_trials=args.n_trials,
        show_progress_bar=False
    )
    
    best_params = study.best_trial.params
    best_val_tune = study.best_trial.value
    
    print(f"[MYEVAE-OPTUNA] Best trial on Fold 0: val_loss={best_val_tune:.4f} | d={best_params['d']} | lambda_kl={best_params['lambda_kl']} | lambda_cox={best_params['lambda_cox']} | lr={best_params['lr']}")
    
    with open(f"{out_path}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"[MYEVAE-OPTUNA] Retraining on all 5 folds with best params...")
    for k, fold_dict in enumerate(folds, 1):
        np.random.seed(42)
        torch.manual_seed(42)
        train_idx, val_idx, test_idx = fold_dict['train'], fold_dict['val'], fold_dict['test']
        
        Xc_tr, Mc_tr = X_c_masked[train_idx].copy(), mask_c_full[train_idx]
        Xg_tr, Mg_tr = X_g_masked[train_idx].copy(), mask_g_full[train_idx]
        Xc_vl, Mc_vl = X_c_masked[val_idx].copy(), mask_c_full[val_idx]
        Xg_vl, Mg_vl = X_g_masked[val_idx].copy(), mask_g_full[val_idx]
        Xc_ts, Mc_ts = X_c_masked[test_idx].copy(), mask_c_full[test_idx]
        Xg_ts, Mg_ts = X_g_masked[test_idx].copy(), mask_g_full[test_idx]
        
        c_means = np.nanmean(Xc_tr, axis=0)
        g_means = np.nanmean(Xg_tr, axis=0)
        c_means[np.isnan(c_means)] = 0.0
        g_means[np.isnan(g_means)] = 0.0
        
        for i in range(p_c):
            Xc_tr[np.isnan(Xc_tr[:, i]), i] = c_means[i]
            Xc_vl[np.isnan(Xc_vl[:, i]), i] = c_means[i]
            Xc_ts[np.isnan(Xc_ts[:, i]), i] = c_means[i]
        for i in range(p_g):
            Xg_tr[np.isnan(Xg_tr[:, i]), i] = g_means[i]
            Xg_vl[np.isnan(Xg_vl[:, i]), i] = g_means[i]
            Xg_ts[np.isnan(Xg_ts[:, i]), i] = g_means[i]
            
        ds_train = TensorDataset(torch.FloatTensor(Xc_tr), torch.FloatTensor(Mc_tr), torch.FloatTensor(Xg_tr), torch.FloatTensor(Mg_tr), torch.FloatTensor(T[train_idx]), torch.FloatTensor(E[train_idx]))
        ds_val = TensorDataset(torch.FloatTensor(Xc_vl), torch.FloatTensor(Mc_vl), torch.FloatTensor(Xg_vl), torch.FloatTensor(Mg_vl), torch.FloatTensor(T[val_idx]), torch.FloatTensor(E[val_idx]))
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
        

        
        np.random.seed(42)
        torch.manual_seed(42)
        
        model = MyeVAE(p_c, p_g, best_params['d']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"{out_path}/fold_{k}_model.pt"
        
        for ep in range(1, args.max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xc_mb, mc_mb, xg_mb, mg_mb, t_mb, e_mb in dl_train:
                xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
                t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                optimizer.zero_grad()
                x_hat_c, x_hat_g, eta, mu_c, log_var_c, mu_g, log_var_g, z_shared = model(xc_mb, mc_mb, xg_mb, mg_mb)
                l_recon = masked_recon_loss_c(x_hat_c, xc_mb, mc_mb) + masked_mse(x_hat_g, xg_mb, mg_mb)
                l_kl = kl_divergence(mu_c, log_var_c) + kl_divergence(mu_g, log_var_g)
                l_cox = cox_partial_likelihood(eta, t_mb, e_mb)
                loss = l_recon + best_params['lambda_kl'] * l_kl + best_params['lambda_cox'] * l_cox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(dl_train)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xc_mb, mc_mb, xg_mb, mg_mb, _, _ in dl_val:
                    xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                    xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
                    x_hat_c, x_hat_g, _, _, _, _, _, _ = model(xc_mb, mc_mb, xg_mb, mg_mb)
                    val_loss += (masked_recon_loss_c(x_hat_c, xc_mb, mc_mb) + masked_mse(x_hat_g, xg_mb, mg_mb)).item()
            val_loss /= len(dl_val)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
                    
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            xc_ts_t, mc_ts_t = torch.FloatTensor(Xc_ts).to(device), torch.FloatTensor(Mc_ts).to(device)
            xg_ts_t, mg_ts_t = torch.FloatTensor(Xg_ts).to(device), torch.FloatTensor(Mg_ts).to(device)
            x_hat_c, x_hat_g, eta, _, _, _, _, z_shared = model(xc_ts_t, mc_ts_t, xg_ts_t, mg_ts_t)
            x_hat_c = x_hat_c.cpu().numpy()
            bin_cols = [1, 2, 3, 4, 5, 6]
            x_hat_c[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c[:, bin_cols]))
            x_hat_g = x_hat_g.cpu().numpy()
            eta = eta.cpu().numpy()
            z = z_shared.cpu().numpy()
            
        obs_c = mask_c_full[test_idx][:, 1:] == 1
        diff_c = x_hat_c[:, 1:][obs_c] - X_c[test_idx][:, 1:][obs_c]
        rmse_c = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
        mae_c = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
        
        obs_g = mask_g_full[test_idx][:, :49] == 1
        diff_g = x_hat_g[:, :49][obs_g] - X_g[test_idx][:, :49][obs_g]
        rmse_g = np.sqrt(np.mean(diff_g**2)) if obs_g.sum() > 0 else 0.0
        mae_g = np.mean(np.abs(diff_g)) if obs_g.sum() > 0 else 0.0
        
        print(f"[MYEVAE-OPTUNA] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f} | RMSE_g: {rmse_g:.4f} | MAE_g: {mae_g:.4f}")
        with open(f"{out_path}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump({'test_idx': test_idx, 'x_hat_c': x_hat_c, 'x_hat_g': x_hat_g, 'eta': eta, 'z': z,
                         'rmse_c': rmse_c, 'mae_c': mae_c, 'rmse_g': rmse_g, 'mae_g': mae_g}, f)
        results.append({'fold': k, 'rmse_c': rmse_c, 'mae_c': mae_c, 'rmse_g': rmse_g, 'mae_g': mae_g, 'best_d': best_params['d']})
        
    df = pd.DataFrame(results)
    mean_rmse_c, std_rmse_c = df['rmse_c'].mean(), df['rmse_c'].std()
    mean_mae_c, std_mae_c = df['mae_c'].mean(), df['mae_c'].std()
    mean_rmse_g, std_rmse_g = df['rmse_g'].mean(), df['rmse_g'].std()
    mean_mae_g, std_mae_g = df['mae_g'].mean(), df['mae_g'].std()
    
    df_summary = pd.DataFrame([{
        'mean_rmse_c': mean_rmse_c, 'std_rmse_c': std_rmse_c,
        'mean_mae_c': mean_mae_c, 'std_mae_c': std_mae_c,
        'mean_rmse_g': mean_rmse_g, 'std_rmse_g': std_rmse_g,
        'mean_mae_g': mean_mae_g, 'std_mae_g': std_mae_g,
        'most_common_d': df['best_d'].mode()[0] if not df['best_d'].empty else np.nan
    }])
    df_summary.to_csv(f"{out_path}/summary.csv", index=False)
    
if __name__ == '__main__':
    main()
