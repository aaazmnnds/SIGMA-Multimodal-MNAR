# Ablation study runner for SIGMA.
# Evaluates 5 ablation variants under severe independent MNAR conditions with Optuna tuning.

import os, pickle, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.special import expit
import optuna

from sigma_model import Encoder, Decoder, SurvivalNet, EarlyFusion
from sigma_model import masked_mse, kl_divergence, cox_partial_likelihood, infonce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BINARY_COLS_C = [1, 2, 3, 4, 5, 6]
CONTINUOUS_COLS_C = [0, 7, 8]

def masked_recon_loss_c(preds, targets, mask):
    obs = (1 - mask).bool()
    losses = []
    cont_idx = torch.tensor(CONTINUOUS_COLS_C, device=preds.device)
    preds_cont = preds[:, cont_idx]
    targets_cont = targets[:, cont_idx]
    obs_cont = obs[:, cont_idx]
    if obs_cont.sum() > 0:
        mse = ((preds_cont - targets_cont) ** 2 * obs_cont).sum() / (obs_cont.sum() + 1e-8)
        losses.append(mse)
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

class SIGMAAblationModel(nn.Module):
    def __init__(self, p_c, p_g, d, variant='full'):
        super().__init__()
        self.p_c = p_c
        self.p_g = p_g
        self.d = d
        self.variant = variant

        # Encoder input dims
        enc_in_c = p_c if variant == 'no_mask' else 2 * p_c
        enc_in_g = p_g if variant == 'no_mask' else 2 * p_g

        if variant != 'genomic_only':
            self.enc_c = Encoder(enc_in_c, d)
            self.dec_c = Decoder(d, p_c)
        if variant != 'clinical_only':
            self.enc_g = Encoder(enc_in_g, d)
            self.dec_g = Decoder(d, p_g)

        if variant not in ['clinical_only', 'genomic_only']:
            self.fusion_module = EarlyFusion(d)

        if variant != 'no_cox':
            self.survival_net = SurvivalNet(d)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x_c, mask_c, x_g, mask_g):
        if self.variant == 'clinical_only':
            in_c = x_c if self.variant == 'no_mask' else torch.cat([x_c, mask_c], dim=-1)
            mu_c, log_var_c = self.enc_c(in_c)
            z_c = self.reparameterize(mu_c, log_var_c)
            z_shared = z_c
            x_hat_c = self.dec_c(z_shared)
            x_hat_g = torch.zeros((x_c.shape[0], self.p_g), device=x_c.device)
            eta = self.survival_net(z_shared)
            return x_hat_c, x_hat_g, eta, z_c, None, z_shared, mu_c, log_var_c

        elif self.variant == 'genomic_only':
            in_g = x_g if self.variant == 'no_mask' else torch.cat([x_g, mask_g], dim=-1)
            mu_g, log_var_g = self.enc_g(in_g)
            z_g = self.reparameterize(mu_g, log_var_g)
            z_shared = z_g
            x_hat_g = self.dec_g(z_shared)
            x_hat_c = torch.zeros((x_g.shape[0], self.p_c), device=x_g.device)
            eta = self.survival_net(z_shared)
            return x_hat_c, x_hat_g, eta, None, z_g, z_shared, mu_g, log_var_g

        else:
            in_c = x_c if self.variant == 'no_mask' else torch.cat([x_c, mask_c], dim=-1)
            in_g = x_g if self.variant == 'no_mask' else torch.cat([x_g, mask_g], dim=-1)

            mu_c, log_var_c = self.enc_c(in_c)
            mu_g, log_var_g = self.enc_g(in_g)

            z_c = self.reparameterize(mu_c, log_var_c)
            z_g = self.reparameterize(mu_g, log_var_g)

            z_shared = self.fusion_module(z_c, z_g)
            mu_shared = (mu_c + mu_g) / 2.0
            log_var_shared = (log_var_c + log_var_g) / 2.0

            x_hat_c = self.dec_c(z_shared)
            x_hat_g = self.dec_g(z_shared)
            eta = self.survival_net(z_shared) if self.variant != 'no_cox' else torch.zeros(x_c.shape[0], device=x_c.device)

            return x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared

class AblationDataset(Dataset):
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
        return (self.x_c[idx], self.mask_c[idx], self.x_g[idx], self.mask_g[idx], self.T[idx], self.E[idx])

def evaluate_ablation(model, dataloader, variant):
    model.eval()
    all_x_hat_c, all_x_hat_g, all_eta = [], [], []
    all_z_shared, all_z_c, all_z_g = [], [], []
    with torch.no_grad():
        for x_c, mask_c, x_g, mask_g, T, E in dataloader:
            x_c, mask_c = x_c.to(device), mask_c.to(device)
            x_g, mask_g = x_g.to(device), mask_g.to(device)
            x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, _, _ = model(x_c, mask_c, x_g, mask_g)
            all_x_hat_c.append(x_hat_c.cpu())
            all_x_hat_g.append(x_hat_g.cpu())
            all_eta.append(eta.cpu())
            all_z_shared.append(z_shared.cpu())
            if z_c is not None:
                all_z_c.append(z_c.cpu())
            if z_g is not None:
                all_z_g.append(z_g.cpu())

    return (
        torch.cat(all_x_hat_c, dim=0),
        torch.cat(all_x_hat_g, dim=0),
        torch.cat(all_eta, dim=0),
        torch.cat(all_z_shared, dim=0),
        torch.cat(all_z_c, dim=0) if all_z_c else None,
        torch.cat(all_z_g, dim=0) if all_z_g else None
    )

def train_ablation_variant(variant, data_dict, folds, out_base_dir, n_trials=50, max_epochs=100, tune_epochs=35, patience=15, batch_size=128):
    print(f"\n=======================================================")
    print(f"  STARTING ABLATION VARIANT: {variant}")
    print(f"=======================================================")
    out_path = f"{out_base_dir}/ablation_{variant}_severe_independent"
    os.makedirs(out_path, exist_ok=True)

    X_c = data_dict['X_c']
    X_g = data_dict['X_g']
    X_expr = data_dict['X_expr']
    T_all = data_dict['T_all']
    E_all = data_dict['E_all']
    mask_c_full = data_dict['mask_c_full']
    mask_g_full = data_dict['mask_g_full']
    X_c_masked = data_dict['X_c_masked']
    X_g_masked = data_dict['X_g_masked']

    p_c = X_c.shape[1]
    p_g = X_g.shape[1]

    # Objective for Fold 0 tuning
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']

    train_ds_0 = AblationDataset(X_c_masked[train_idx_0], X_g_masked[train_idx_0], T_all[train_idx_0], E_all[train_idx_0], mask_c_full[train_idx_0], mask_g_full[train_idx_0])
    val_ds_0 = AblationDataset(X_c_masked[val_idx_0], X_g_masked[val_idx_0], T_all[val_idx_0], E_all[val_idx_0], mask_c_full[val_idx_0], mask_g_full[val_idx_0])

    train_loader_0 = DataLoader(train_ds_0, batch_size=batch_size, shuffle=True)
    val_loader_0 = DataLoader(val_ds_0, batch_size=batch_size, shuffle=False)

    def optuna_objective(trial):
        d = trial.suggest_categorical('d', [32, 64, 128])
        lambda_kl = trial.suggest_float('lambda_kl', 1e-2, 2.0, log=True)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        if variant == 'no_cox':
            lambda_cox = 0.0
        else:
            lambda_cox = trial.suggest_float('lambda_cox', 0.1, 5.0, log=True)

        if variant in ['no_cross', 'clinical_only', 'genomic_only']:
            lambda_cross = 0.0
            kappa = 0.1
        else:
            lambda_cross = trial.suggest_float('lambda_cross', 0.1, 1.0, log=True)
            kappa = trial.suggest_float('kappa', 0.05, 0.5, log=True)

        model = SIGMAAblationModel(p_c, p_g, d, variant).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val = float('inf')
        patience_count = 0

        for epoch in range(tune_epochs):
            model.train()
            for x_c, mask_c, x_g, mask_g, T, E in train_loader_0:
                x_c, mask_c = x_c.to(device), mask_c.to(device)
                x_g, mask_g = x_g.to(device), mask_g.to(device)
                T, E = T.to(device), E.to(device)

                optimizer.zero_grad()
                x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(x_c, mask_c, x_g, mask_g)

                loss = 0.0
                if variant != 'genomic_only':
                    loss += masked_recon_loss_c(x_hat_c, x_c, mask_c)
                if variant != 'clinical_only':
                    loss += masked_mse(x_hat_g, x_g, mask_g)

                loss += lambda_kl * kl_divergence(mu_shared, log_var_shared)

                if variant != 'no_cox':
                    loss += lambda_cox * cox_partial_likelihood(eta, T, E)

                if variant not in ['no_cross', 'clinical_only', 'genomic_only'] and z_c is not None and z_g is not None:
                    loss += lambda_cross * infonce_loss(z_c, z_g, kappa)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_recon_loss = 0.0
            all_eta_val = []
            all_T_val, all_E_val = [], []
            with torch.no_grad():
                for x_c, mask_c, x_g, mask_g, T, E in val_loader_0:
                    x_c, mask_c = x_c.to(device), mask_c.to(device)
                    x_g, mask_g = x_g.to(device), mask_g.to(device)
                    x_hat_c, x_hat_g, eta, _, _, _, _, _ = model(x_c, mask_c, x_g, mask_g)
                    
                    if variant != 'genomic_only':
                        val_recon_loss += masked_recon_loss_c(x_hat_c, x_c, mask_c).item()
                    if variant != 'clinical_only':
                        val_recon_loss += masked_mse(x_hat_g, x_g, mask_g).item()

                    if variant != 'no_cox':
                        all_eta_val.append(eta.cpu().numpy().flatten())
                        all_T_val.append(T.cpu().numpy())
                        all_E_val.append(E.cpu().numpy())

            val_recon_loss /= len(val_loader_0)

            if variant == 'no_cox':
                val_loss = val_recon_loss
            else:
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(optuna_objective, n_trials=n_trials)

    best_params = study.best_params
    if 'lambda_cox' not in best_params:
        best_params['lambda_cox'] = 0.0
    if 'lambda_cross' not in best_params:
        best_params['lambda_cross'] = 0.0
    if 'kappa' not in best_params:
        best_params['kappa'] = 0.1

    print(f"[{variant}] Optuna Best Params: {best_params}")
    with open(f"{out_path}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)

    # ==== RETRAIN ACROSS 5 FOLDS ====
    results = []
    for k, fold in enumerate(folds, 1):
        train_idx = fold['train']
        val_idx = fold['val']
        test_idx = fold['test']

        train_ds = AblationDataset(X_c_masked[train_idx], X_g_masked[train_idx], T_all[train_idx], E_all[train_idx], mask_c_full[train_idx], mask_g_full[train_idx])
        val_ds = AblationDataset(X_c_masked[val_idx], X_g_masked[val_idx], T_all[val_idx], E_all[val_idx], mask_c_full[val_idx], mask_g_full[val_idx])
        test_ds = AblationDataset(X_c_masked[test_idx], X_g_masked[test_idx], T_all[test_idx], E_all[test_idx], mask_c_full[test_idx], mask_g_full[test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = SIGMAAblationModel(p_c, p_g, best_params['d'], variant).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

        best_val_loss = float('inf')
        patience_count = 0
        best_model_path = f"{out_path}/fold_{k}_model.pt"

        for epoch in range(1, max_epochs + 1):
            model.train()
            for x_c, mask_c, x_g, mask_g, T, E in train_loader:
                x_c, mask_c = x_c.to(device), mask_c.to(device)
                x_g, mask_g = x_g.to(device), mask_g.to(device)
                T, E = T.to(device), E.to(device)

                optimizer.zero_grad()
                x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(x_c, mask_c, x_g, mask_g)

                loss = 0.0
                if variant != 'genomic_only':
                    loss += masked_recon_loss_c(x_hat_c, x_c, mask_c)
                if variant != 'clinical_only':
                    loss += masked_mse(x_hat_g, x_g, mask_g)

                loss += best_params['lambda_kl'] * kl_divergence(mu_shared, log_var_shared)

                if variant != 'no_cox':
                    loss += best_params['lambda_cox'] * cox_partial_likelihood(eta, T, E)

                if variant not in ['no_cross', 'clinical_only', 'genomic_only'] and z_c is not None and z_g is not None:
                    loss += best_params['lambda_cross'] * infonce_loss(z_c, z_g, best_params['kappa'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_c, mask_c, x_g, mask_g, T, E in val_loader:
                    x_c, mask_c = x_c.to(device), mask_c.to(device)
                    x_g, mask_g = x_g.to(device), mask_g.to(device)
                    x_hat_c, x_hat_g, _, _, _, _, _, _ = model(x_c, mask_c, x_g, mask_g)
                    if variant != 'genomic_only':
                        val_loss += masked_recon_loss_c(x_hat_c, x_c, mask_c).item()
                    if variant != 'clinical_only':
                        val_loss += masked_mse(x_hat_g, x_g, mask_g).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_count += 1
            if patience_count >= patience:
                break

        # Test evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        x_hat_c_test, x_hat_g_test, eta_test, z_shared_test, z_c_test, z_g_test = evaluate_ablation(model, test_loader, variant)

        x_c_test_true = X_c[test_idx]
        x_g_expr_test_true = X_expr[test_idx]
        mask_c_test_features = mask_c_full[test_idx][:, 1:]
        mask_g_expr_test = mask_g_full[test_idx][:, :49]

        x_hat_c_np = x_hat_c_test.numpy()
        bin_cols = [1, 2, 3, 4, 5, 6]
        x_hat_c_np[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c_np[:, bin_cols]))

        rmse_c, mae_c = 0.0, 0.0
        if variant != 'genomic_only':
            x_hat_c_eval = x_hat_c_np[:, 1:]
            x_c_eval_true = x_c_test_true[:, 1:]
            obs_c = mask_c_test_features == 1
            if obs_c.sum() > 0:
                diff_c = x_hat_c_eval[obs_c] - x_c_eval_true[obs_c]
                rmse_c = np.sqrt(np.mean(diff_c**2))
                mae_c = np.mean(np.abs(diff_c))

        rmse_g, mae_g = 0.0, 0.0
        if variant != 'clinical_only':
            x_hat_g_expr = x_hat_g_test.numpy()[:, :49]
            obs_g = mask_g_expr_test == 1
            if obs_g.sum() > 0:
                diff_g = x_hat_g_expr[obs_g] - x_g_expr_test_true[obs_g]
                rmse_g = np.sqrt(np.mean(diff_g**2))
                mae_g = np.mean(np.abs(diff_g))

        print(f"[{variant}] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | RMSE_g: {rmse_g:.4f}")

        res_dict = {
            'x_hat_c': x_hat_c_np,
            'x_hat_g': x_hat_g_test.numpy(),
            'eta': eta_test.numpy(),
            'z_shared': z_shared_test.numpy(),
            'z_c': z_c_test.numpy() if z_c_test is not None else None,
            'z_g': z_g_test.numpy() if z_g_test is not None else None,
            'test_idx': test_idx,
            'rmse_c': rmse_c,
            'mae_c': mae_c,
            'rmse_g': rmse_g,
            'mae_g': mae_g
        }
        with open(f"{out_path}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump(res_dict, f)

        results.append({
            'fold': k,
            'rmse_c': rmse_c,
            'mae_c': mae_c,
            'rmse_g': rmse_g,
            'mae_g': mae_g,
            'best_d': best_params['d']
        })

    df = pd.DataFrame(results)
    mean_rmse_c, std_rmse_c = df['rmse_c'].mean(), df['rmse_c'].std()
    mean_mae_c, std_mae_c = df['mae_c'].mean(), df['mae_c'].std()
    mean_rmse_g, std_rmse_g = df['rmse_g'].mean(), df['rmse_g'].std()
    mean_mae_g, std_mae_g = df['mae_g'].mean(), df['mae_g'].std()

    print(f"[{variant}] Summary | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | RMSE_g: {mean_rmse_g:.4f} ± {std_rmse_g:.4f}")

    summary_data = [{
        'mean_rmse_c': mean_rmse_c, 'std_rmse_c': std_rmse_c,
        'mean_mae_c': mean_mae_c, 'std_mae_c': std_mae_c,
        'mean_rmse_g': mean_rmse_g, 'std_rmse_g': std_rmse_g,
        'mean_mae_g': mean_mae_g, 'std_mae_g': std_mae_g,
        'most_common_d': df['best_d'].mode()[0]
    }]
    pd.DataFrame(summary_data).to_csv(f"{out_path}/summary.csv", index=False)

def main():
    BASE = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    FOLDS_DIR = f"{PROCESSED_DIR}/folds"
    OUTPUT_DIR = f"{BASE}/results_final"

    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")

    with open(f"{FOLDS_DIR}/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)

    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    genomic_cols = genomic.drop(columns=['PATIENT_ID']).columns.tolist()
    expr_cols = [c for c in genomic_cols if not c.endswith('_CNA')]
    X_expr = genomic[expr_cols].values.astype(np.float32)

    T_all = survival['T'].values.astype(np.float32)
    E_all = survival['E'].values.astype(np.float32)

    masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
    masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")

    # Severe independent condition (index 2)
    severity_idx = 2
    mask_c_features = masks_c[severity_idx]
    age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)

    mask_g_expr_full = masks_g_expr[severity_idx]

    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0

    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = 0.0

    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full

    data_dict = {
        'X_c': X_c, 'X_g': X_g, 'X_expr': X_expr,
        'T_all': T_all, 'E_all': E_all,
        'mask_c_full': mask_c_full, 'mask_g_full': mask_g_full,
        'X_c_masked': X_c_masked, 'X_g_masked': X_g_masked
    }

    variants = ['no_cross', 'no_cox', 'no_mask', 'clinical_only', 'genomic_only']
    for v in variants:
        train_ablation_variant(v, data_dict, folds, OUTPUT_DIR, n_trials=50, max_epochs=100, tune_epochs=35, patience=15)

if __name__ == '__main__':
    main()
