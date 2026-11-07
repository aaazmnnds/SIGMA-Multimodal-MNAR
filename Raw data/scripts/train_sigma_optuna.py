# SIGMA training script with Optuna nested cross-validation hyperparameter tuning.

import os, pickle, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.special import expit
import optuna
from sigma_model import SIGMA, SIGMADataset, train_sigma, evaluate_sigma
from sigma_model import masked_mse, kl_divergence, cox_partial_likelihood, infonce_loss, device


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


def objective(trial, train_loader, val_loader, p_c, p_g, fusion, batch_size, tune_epochs, device):
    d = trial.suggest_categorical('d', [32, 64, 128])
    lambda_kl = trial.suggest_float('lambda_kl', 1e-2, 2.0, log=True)
    lambda_cox = trial.suggest_float('lambda_cox', 0.1, 5.0, log=True)
    lambda_cross = trial.suggest_float('lambda_cross', 0.1, 1.0, log=True)
    kappa = trial.suggest_float('kappa', 0.05, 0.5, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    model = SIGMA(p_c, p_g, d, fusion).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val = float('inf')
    patience_count = 0
    
    for epoch in range(tune_epochs):
        # training step
        model.train()
        for x_c, mask_c, x_g, mask_g, T, E in train_loader:
            x_c, mask_c = x_c.to(device), mask_c.to(device)
            x_g, mask_g = x_g.to(device), mask_g.to(device)
            T, E = T.to(device), E.to(device)
            
            optimizer.zero_grad()
            x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(x_c, mask_c, x_g, mask_g)
            
            l_c = masked_recon_loss_c(x_hat_c, x_c, mask_c)
            l_g = masked_mse(x_hat_g, x_g, mask_g)
            l_kl = kl_divergence(mu_shared, log_var_shared)
            l_cox = cox_partial_likelihood(eta, T, E)
            l_cross = infonce_loss(z_c, z_g, kappa)
            
            loss = l_c + l_g + lambda_kl * l_kl + lambda_cox * l_cox + lambda_cross * l_cross
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # validation reconstruction loss
        model.eval()
        val_recon_loss = 0.0
        all_eta_val = []
        all_T_val = []
        all_E_val = []
        with torch.no_grad():
            for x_c, mask_c, x_g, mask_g, T, E in val_loader:
                x_c, mask_c = x_c.to(device), mask_c.to(device)
                x_g, mask_g = x_g.to(device), mask_g.to(device)
                T, E = T.to(device), E.to(device)
                x_hat_c, x_hat_g, eta, _, _, _, _, _ = model(x_c, mask_c, x_g, mask_g)
                l_c = masked_recon_loss_c(x_hat_c, x_c, mask_c)
                l_g = masked_mse(x_hat_g, x_g, mask_g)
                val_recon_loss += (l_c + l_g).item()
                all_eta_val.append(eta.cpu().numpy().flatten())
                all_T_val.append(T.cpu().numpy())
                all_E_val.append(E.cpu().numpy())
        val_recon_loss /= len(val_loader)
        
        # Compute C-index on validation set
        try:
            from sksurv.metrics import concordance_index_censored
            eta_val = np.concatenate(all_eta_val)
            T_val = np.concatenate(all_T_val)
            E_val = np.concatenate(all_E_val)
            eta_val = np.nan_to_num(eta_val, nan=0.0, posinf=1e6, neginf=-1e6)
            val_c_index = concordance_index_censored(E_val.astype(bool), T_val, eta_val)[0]
        except:
            val_c_index = 0.5
        
        # Composite objective: minimize recon loss, maximize C-index
        lambda_surv = 2.0
        val_loss = val_recon_loss - lambda_surv * val_c_index
        
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
        else:
            patience_count += 1
        
        if patience_count >= 10:
            break
        
        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion', type=str, required=True, choices=['EF', 'CMA', 'PoE'])
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--mechanism', type=str, choices=['threshold', 'selfmask'], default='threshold')
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
    if args.mechanism == 'selfmask':
        OUTPUT_DIR = f"{BASE}/results_selfmask"
    else:
        OUTPUT_DIR = f"{BASE}/results_final"
    
    out_path = f"{OUTPUT_DIR}/{args.fusion}_{args.severity}_{args.condition}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(f"[SIGMA-OPTUNA] Config: fusion={args.fusion} | severity={args.severity} | condition={args.condition} | mechanism={args.mechanism}")

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

    if args.mechanism == 'selfmask':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_clinical.npy")
        masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_genomic.npy")
    else:
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
    
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]
    
    mask_c_features = masks_c[severity_idx]
    age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    
    mask_g_expr_full = masks_g_expr[severity_idx]
    
    if args.condition == 'joint':
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
        X_c_features_arr = X_c[:, 1:]
        if args.mechanism == 'selfmask':
            prob_c = expit(alpha_c * np.abs(X_c_features_arr) + beta_c)
            prob_g = expit(alpha_g * np.abs(X_expr) + beta_g)
        else:
            age = X_c[:, 0]
            tau = np.median(age)
            age_indicator = (age > tau).astype(np.float32)
            prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features_arr) + beta_c)
            prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)

        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
        mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
        
        mask_g_expr_joint = np.random.binomial(1, prob_g).astype(np.float32)
        mask_g_expr_full = mask_g_expr_joint

    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = 0.0
    
    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full
    
    p_c = X_c.shape[1]
    p_g = X_g.shape[1]

    results = []


    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ==== TUNE ON FOLD 0 ====
    print(f"[SIGMA-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    train_dataset_0 = SIGMADataset(X_c_masked[train_idx_0], X_g_masked[train_idx_0], 
                                 T_all[train_idx_0], E_all[train_idx_0], 
                                 mask_c_full[train_idx_0], mask_g_full[train_idx_0])
    val_dataset_0 = SIGMADataset(X_c_masked[val_idx_0], X_g_masked[val_idx_0], 
                               T_all[val_idx_0], E_all[val_idx_0], 
                               mask_c_full[val_idx_0], mask_g_full[val_idx_0])
                                
    train_loader_0 = DataLoader(train_dataset_0, batch_size=args.batch_size, shuffle=True)
    val_loader_0 = DataLoader(val_dataset_0, batch_size=args.batch_size, shuffle=False)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )
    study.optimize(
        lambda trial: objective(trial, train_loader_0, val_loader_0, p_c, p_g, 
                               args.fusion, args.batch_size, args.tune_epochs, device),
        n_trials=args.n_trials,
        show_progress_bar=False
    )
    
    best_params = study.best_trial.params
    best_val_tune = study.best_trial.value
    
    print(f"[SIGMA-OPTUNA] Best trial on Fold 0: val_loss={best_val_tune:.4f} | d={best_params['d']} | lambda_kl={best_params['lambda_kl']} | lambda_cox={best_params['lambda_cox']} | lambda_cross={best_params['lambda_cross']} | kappa={best_params['kappa']} | lr={best_params['lr']}")
    
    with open(f"{out_path}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"[SIGMA-OPTUNA] Retraining on all 5 folds with best params...")

    for k, fold_dict in enumerate(folds, 1):
        train_idx = fold_dict['train']
        val_idx = fold_dict['val']
        test_idx = fold_dict['test']
        
        train_dataset = SIGMADataset(X_c_masked[train_idx], X_g_masked[train_idx], 
                                     T_all[train_idx], E_all[train_idx], 
                                     mask_c_full[train_idx], mask_g_full[train_idx])
        val_dataset = SIGMADataset(X_c_masked[val_idx], X_g_masked[val_idx], 
                                   T_all[val_idx], E_all[val_idx], 
                                   mask_c_full[val_idx], mask_g_full[val_idx])
        test_dataset = SIGMADataset(X_c_masked[test_idx], X_g_masked[test_idx],
                                    T_all[test_idx], E_all[test_idx],
                                    mask_c_full[test_idx], mask_g_full[test_idx])
                                    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SIGMA(p_c, p_g, best_params['d'], args.fusion).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        
        best_val_loss = float('inf')
        patience_count = 0
        best_model_path = f"{out_path}/fold_{k}_model.pt"
        
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            train_loss = 0.0
            for x_c, mask_c, x_g, mask_g, T, E in train_loader:
                x_c, mask_c = x_c.to(device), mask_c.to(device)
                x_g, mask_g = x_g.to(device), mask_g.to(device)
                T, E = T.to(device), E.to(device)
                
                optimizer.zero_grad()
                x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(x_c, mask_c, x_g, mask_g)
                
                l_c = masked_recon_loss_c(x_hat_c, x_c, mask_c)
                l_g = masked_mse(x_hat_g, x_g, mask_g)
                l_kl = kl_divergence(mu_shared, log_var_shared)
                l_cox = cox_partial_likelihood(eta, T, E)
                l_cross = infonce_loss(z_c, z_g, best_params['kappa'])
                
                loss = l_c + l_g + best_params['lambda_kl'] * l_kl + best_params['lambda_cox'] * l_cox + best_params['lambda_cross'] * l_cross
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_c, mask_c, x_g, mask_g, T, E in val_loader:
                    x_c, mask_c = x_c.to(device), mask_c.to(device)
                    x_g, mask_g = x_g.to(device), mask_g.to(device)
                    x_hat_c, x_hat_g, _, _, _, _, _, _ = model(x_c, mask_c, x_g, mask_g)
                    l_c = masked_recon_loss_c(x_hat_c, x_c, mask_c)
                    l_g = masked_mse(x_hat_g, x_g, mask_g)
                    val_loss += (l_c + l_g).item()
            val_loss /= len(val_loader)
            
            if (epoch == 10) or (epoch % 50 == 0) or (epoch == args.max_epochs):
                print(f"[SIGMA-OPTUNA] Fold {k}/5 | Epoch {epoch}/{args.max_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_count += 1
                
            if patience_count >= args.patience:
                print(f"[SIGMA-OPTUNA] Fold {k}/5 | Early stopping at epoch {epoch}")
                break
                
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        
        x_hat_c_test, x_hat_g_test, eta_test, z_shared_test, z_c_test, z_g_test = evaluate_sigma(model, test_loader)
        
        x_c_test_true = X_c[test_idx]
        x_g_expr_test_true = X_expr[test_idx]
        mask_c_test_features = mask_c_full[test_idx][:, 1:]
        mask_g_expr_test = mask_g_full[test_idx][:, :49]
        
        x_hat_c_np = x_hat_c_test.numpy()
        bin_cols = [1, 2, 3, 4, 5, 6]
        x_hat_c_np[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c_np[:, bin_cols]))
        x_hat_c_features = x_hat_c_np[:, 1:]
        x_c_test_features = x_c_test_true[:, 1:]
        obs_c = mask_c_test_features == 1
        if obs_c.sum() > 0:
            diff_c = x_hat_c_features[obs_c] - x_c_test_features[obs_c]
            rmse_c = np.sqrt(np.mean(diff_c**2))
            mae_c = np.mean(np.abs(diff_c))
        else:
            rmse_c, mae_c = 0.0, 0.0
            
        x_hat_g_expr = x_hat_g_test.numpy()[:, :49]
        obs_g = mask_g_expr_test == 1
        if obs_g.sum() > 0:
            diff_g = x_hat_g_expr[obs_g] - x_g_expr_test_true[obs_g]
            rmse_g = np.sqrt(np.mean(diff_g**2))
            mae_g = np.mean(np.abs(diff_g))
        else:
            rmse_g, mae_g = 0.0, 0.0
            
        print(f"[SIGMA-OPTUNA] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f} | RMSE_g: {rmse_g:.4f} | MAE_g: {mae_g:.4f}")
        
        res_dict = {
            'x_hat_c': x_hat_c_np,
            'x_hat_g': x_hat_g_test.numpy(),
            'eta': eta_test.numpy(),
            'z_shared': z_shared_test.numpy(),
            'z_c': z_c_test.numpy(),
            'z_g': z_g_test.numpy(),
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
    
    print(f"[SIGMA-OPTUNA] Summary | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | MAE_c: {mean_mae_c:.4f} ± {std_mae_c:.4f} | RMSE_g: {mean_rmse_g:.4f} ± {std_rmse_g:.4f} | MAE_g: {mean_mae_g:.4f} ± {std_mae_g:.4f}")
    
    summary_data = [{
        'mean_rmse_c': mean_rmse_c, 'std_rmse_c': std_rmse_c,
        'mean_mae_c': mean_mae_c, 'std_mae_c': std_mae_c,
        'mean_rmse_g': mean_rmse_g, 'std_rmse_g': std_rmse_g,
        'mean_mae_g': mean_mae_g, 'std_mae_g': std_mae_g,
        'most_common_d': df['best_d'].mode()[0]
    }]
    pd.DataFrame(summary_data).to_csv(f"{out_path}/summary.csv", index=False)



if __name__ == '__main__':
    main()
