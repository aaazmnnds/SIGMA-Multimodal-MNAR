# Deep learning baseline script for SIGMA study. Implements StandardVAE, MNARSVAE, and SIGMA ablations.
import os, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.special import expit

from sigma_model import SIGMA, SIGMADataset, train_sigma, evaluate_sigma
from sigma_model import Encoder, Decoder, SurvivalNet
from sigma_model import masked_mse, kl_divergence, cox_partial_likelihood, device

class StandardVAE(nn.Module):
    def __init__(self, p_c, d):
        super().__init__()
        self.encoder = Encoder(p_c * 2, d)
        self.decoder = Decoder(d, p_c)
        
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
        return x_hat_c, mu, log_var, z

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['StandardVAE', 'MNARSVAE', 'SIGMA_clinical', 'SIGMA_genomic'])
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--fusion', type=str, default='PoE', choices=['EF', 'CMA', 'PoE'])
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_cox', type=float, default=1.0)
    parser.add_argument('--lambda_cross', type=float, default=0.5)
    parser.add_argument('--kappa', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DIR = f"{BASE}/processed"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    FOLDS_DIR = f"{PROCESSED_DIR}/folds"
    OUTPUT_DIR = f"{BASE}/results"
    
    if args.model in ['SIGMA_clinical', 'SIGMA_genomic']:
        out_path = f"{OUTPUT_DIR}/{args.model}_{args.fusion}_{args.severity}_{args.condition}"
    else:
        out_path = f"{OUTPUT_DIR}/{args.model}_{args.severity}_{args.condition}"
        
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    print(f"[DEEP_BASELINES] Config: model={args.model} | severity={args.severity} | condition={args.condition}")
    
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
    X_c_masked[mask_c_full == 1] = 0.0
    
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = 0.0
    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full
    
    p_c = X_c.shape[1]
    p_g = X_g.shape[1]
    
    results = []
    
    for k, fold_dict in enumerate(folds, 1):
        np.random.seed(42)
        torch.manual_seed(42)
        train_idx, val_idx, test_idx = fold_dict['train'], fold_dict['val'], fold_dict['test']
        
        if args.model in ['StandardVAE', 'MNARSVAE']:
            model = StandardVAE(p_c, args.d).to(device) if args.model == 'StandardVAE' else MNARSVAEModel(p_c, args.d).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            ds_train = SIGMADataset(X_c_masked[train_idx], X_g_masked[train_idx], T[train_idx], E[train_idx], mask_c_full[train_idx], mask_g_full[train_idx])
            ds_val = SIGMADataset(X_c_masked[val_idx], X_g_masked[val_idx], T[val_idx], E[val_idx], mask_c_full[val_idx], mask_g_full[val_idx])
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
            dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_path = f"{out_path}/fold_{k}_model.pt"
            
            for ep in range(1, args.max_epochs + 1):
                model.train()
                train_loss = 0.0
                for xc_mb, mc_mb, _, _, t_mb, e_mb in dl_train:
                    xc_mb, mc_mb, t_mb, e_mb = xc_mb.to(device), mc_mb.to(device), t_mb.to(device), e_mb.to(device)
                    
                    if args.model == 'StandardVAE':
                        x_hat_c, mu, log_var, z = model(xc_mb, mc_mb)
                        recon_loss = masked_mse(x_hat_c, xc_mb, mc_mb)
                        kl_loss = kl_divergence(mu, log_var)
                        loss = recon_loss + args.lambda_kl * kl_loss
                    else:
                        x_hat_c, mu, log_var, z, eta = model(xc_mb, mc_mb)
                        recon_loss = masked_mse(x_hat_c, xc_mb, mc_mb)
                        kl_loss = kl_divergence(mu, log_var)
                        cox_loss = cox_partial_likelihood(eta, t_mb, e_mb)
                        loss = recon_loss + args.lambda_kl * kl_loss + args.lambda_cox * cox_loss
                        
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * xc_mb.size(0)
                    
                train_loss /= len(ds_train)
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xc_mb, mc_mb, _, _, _, _ in dl_val:
                        xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                        if args.model == 'StandardVAE':
                            x_hat_c, _, _, _ = model(xc_mb, mc_mb)
                        else:
                            x_hat_c, _, _, _, _ = model(xc_mb, mc_mb)
                        val_loss += masked_mse(x_hat_c, xc_mb, mc_mb).item() * xc_mb.size(0)
                val_loss /= len(ds_val)
                
                if ep % 10 == 0:
                    print(f"[DEEP_BASELINES] Fold {k}/5 | Epoch {ep}/{args.max_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"[DEEP_BASELINES] Fold {k}/5 | Early stopping at epoch {ep}")
                        break
                        
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            with torch.no_grad():
                xc_ts, mc_ts = torch.FloatTensor(X_c_masked[test_idx]).to(device), torch.FloatTensor(mask_c_full[test_idx]).to(device)
                if args.model == 'StandardVAE':
                    x_hat_c, _, _, z = model(xc_ts, mc_ts)
                    eta = torch.full((x_hat_c.size(0), 1), float('nan')).to(device)
                else:
                    x_hat_c, _, _, z, eta = model(xc_ts, mc_ts)
                x_hat_c = x_hat_c.cpu().numpy()
                z = z.cpu().numpy()
                eta = eta.cpu().numpy()
                x_hat_g = np.full_like(X_g[test_idx], np.nan)
                
            obs_c = mask_c_full[test_idx][:, 1:] == 1
            diff_c = x_hat_c[:, 1:][obs_c] - X_c[test_idx][:, 1:][obs_c]
            rmse_c = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
            mae_c = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
            rmse_g, mae_g = float('nan'), float('nan')
            
        else: # SIGMA ablations
            model = SIGMA(p_c, p_g, args.d, args.fusion).to(device)
            if args.model == 'SIGMA_clinical':
                X_g_train = np.zeros((len(train_idx), p_g))
                M_g_train = np.zeros((len(train_idx), p_g))
                X_g_val = np.zeros((len(val_idx), p_g))
                M_g_val = np.zeros((len(val_idx), p_g))
                X_g_test = np.zeros((len(test_idx), p_g))
                M_g_test = np.zeros((len(test_idx), p_g))
                X_c_train, M_c_train = X_c_masked[train_idx], mask_c_full[train_idx]
                X_c_val, M_c_val = X_c_masked[val_idx], mask_c_full[val_idx]
                X_c_test, M_c_test = X_c_masked[test_idx], mask_c_full[test_idx]
            else: # SIGMA_genomic
                X_c_train = np.zeros((len(train_idx), p_c))
                M_c_train = np.zeros((len(train_idx), p_c))
                X_c_val = np.zeros((len(val_idx), p_c))
                M_c_val = np.zeros((len(val_idx), p_c))
                X_c_test = np.zeros((len(test_idx), p_c))
                M_c_test = np.zeros((len(test_idx), p_c))
                X_g_train, M_g_train = X_g_masked[train_idx], mask_g_full[train_idx]
                X_g_val, M_g_val = X_g_masked[val_idx], mask_g_full[val_idx]
                X_g_test, M_g_test = X_g_masked[test_idx], mask_g_full[test_idx]
                
            ds_train = SIGMADataset(X_c_train, X_g_train, T[train_idx], E[train_idx], M_c_train, M_g_train)
            ds_val = SIGMADataset(X_c_val, X_g_val, T[val_idx], E[val_idx], M_c_val, M_g_val)
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
            dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
            
            best_model_path = f"{out_path}/fold_{k}_model.pt"
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            for ep in range(1, args.max_epochs + 1):
                model.train()
                train_loss = 0.0
                for xc_mb, mc_mb, xg_mb, mg_mb, t_mb, e_mb in dl_train:
                    xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                    xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
                    t_mb, e_mb = t_mb.to(device), e_mb.to(device)
                    optimizer.zero_grad()
                    x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(xc_mb, mc_mb, xg_mb, mg_mb)
                    l_recon = masked_mse(x_hat_c, xc_mb, mc_mb) + masked_mse(x_hat_g, xg_mb, mg_mb)
                    l_kl = kl_divergence(mu_shared, log_var_shared)
                    l_cox = cox_partial_likelihood(eta, t_mb, e_mb)
                    loss = l_recon + args.lambda_kl * l_kl + args.lambda_cox * l_cox
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(dl_train)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xc_mb, mc_mb, xg_mb, mg_mb, t_mb, e_mb in dl_val:
                        xc_mb, mc_mb = xc_mb.to(device), mc_mb.to(device)
                        xg_mb, mg_mb = xg_mb.to(device), mg_mb.to(device)
                        x_hat_c, x_hat_g, _, _, _, _, _, _ = model(xc_mb, mc_mb, xg_mb, mg_mb)
                        val_loss += (masked_mse(x_hat_c, xc_mb, mc_mb) + masked_mse(x_hat_g, xg_mb, mg_mb)).item()
                val_loss /= len(dl_val)
                if ep % 10 == 0:
                    print(f"[DEEP_BASELINES] Fold {k}/5 | Epoch {ep}/{args.max_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"[DEEP_BASELINES] Fold {k}/5 | Early stopping at epoch {ep}")
                        break
            
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            with torch.no_grad():
                xc_ts, mc_ts = torch.FloatTensor(X_c_test).to(device), torch.FloatTensor(M_c_test).to(device)
                xg_ts, mg_ts = torch.FloatTensor(X_g_test).to(device), torch.FloatTensor(M_g_test).to(device)
                x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(xc_ts, mc_ts, xg_ts, mg_ts)
                x_hat_c = x_hat_c.cpu().numpy()
                x_hat_g = x_hat_g.cpu().numpy()
                z = z_shared.cpu().numpy()
                eta = eta.cpu().numpy()
                
            if args.model == 'SIGMA_clinical':
                obs_c = mask_c_full[test_idx][:, 1:] == 1
                diff_c = x_hat_c[:, 1:][obs_c] - X_c[test_idx][:, 1:][obs_c]
                rmse_c = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
                mae_c = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
                rmse_g, mae_g = float('nan'), float('nan')
                x_hat_g = np.full_like(X_g[test_idx], np.nan)
            else:
                obs_g = mask_g_full[test_idx][:, :49] == 1
                diff_g = x_hat_g[:, :49][obs_g] - X_g[test_idx][:, :49][obs_g]
                rmse_g = np.sqrt(np.mean(diff_g**2)) if obs_g.sum() > 0 else 0.0
                mae_g = np.mean(np.abs(diff_g)) if obs_g.sum() > 0 else 0.0
                rmse_c, mae_c = float('nan'), float('nan')
                x_hat_c = np.full_like(X_c[test_idx], np.nan)
                
        def fmt(v):
            return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "nan"
        print(f"[DEEP_BASELINES] Fold {k}/5 | RMSE_c: {fmt(rmse_c)} | MAE_c: {fmt(mae_c)} | RMSE_g: {fmt(rmse_g)} | MAE_g: {fmt(mae_g)}")
        with open(f"{out_path}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump({'x_hat_c': x_hat_c, 'x_hat_g': x_hat_g, 'eta': eta, 'z': z, 'test_idx': test_idx,
                         'rmse_c': rmse_c, 'mae_c': mae_c, 'rmse_g': rmse_g, 'mae_g': mae_g}, f)
        results.append({'fold': k, 'rmse_c': rmse_c, 'mae_c': mae_c, 'rmse_g': rmse_g, 'mae_g': mae_g})
        
    df = pd.DataFrame(results)
    mean_rmse_c = df['rmse_c'].mean()
    std_rmse_c = df['rmse_c'].std()
    mean_mae_c = df['mae_c'].mean()
    std_mae_c = df['mae_c'].std()
    
    mean_rmse_g = df['rmse_g'].mean()
    std_rmse_g = df['rmse_g'].std()
    mean_mae_g = df['mae_g'].mean()
    std_mae_g = df['mae_g'].std()
    
    print(f"[DEEP_BASELINES] Summary | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | MAE_c: {mean_mae_c:.4f} ± {std_mae_c:.4f}")
    df_summary = pd.DataFrame([{
        'mean_rmse_c': mean_rmse_c, 'std_rmse_c': std_rmse_c,
        'mean_mae_c': mean_mae_c, 'std_mae_c': std_mae_c,
        'mean_rmse_g': mean_rmse_g, 'std_rmse_g': std_rmse_g,
        'mean_mae_g': mean_mae_g, 'std_mae_g': std_mae_g
    }])
    df_summary.to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
