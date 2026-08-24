import re

injection_code = """
BINARY_COLS_C = [1, 2, 3, 4, 5, 6]  # CHEMOTHERAPY, HORMONE_THERAPY, RADIO_THERAPY, ER_STATUS, HER2_STATUS, PR_STATUS
CONTINUOUS_COLS_C = [0, 7, 8]  # AGE_AT_DIAGNOSIS, MKI67, EGFR

def masked_recon_loss_c(preds, targets, mask):
    \"\"\"Mixed MSE + BCE reconstruction loss for clinical features.\"\"\"
    obs = (1 - mask).bool()
    loss = torch.tensor(0.0, device=preds.device)
    n_obs = obs.sum()
    if n_obs == 0:
        return loss
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
"""

def patch_file(filepath, is_sigma=False, is_dl=False):
    with open(filepath, 'r') as f:
        content = f.read()
        
    if is_sigma:
        # Inject after imports
        content = re.sub(
            r'(from sigma_model import masked_mse, kl_divergence, cox_partial_likelihood, infonce_loss, device\n)',
            r'\1\n' + injection_code + '\n',
            content
        )
        content = content.replace("masked_mse(x_hat_c, x_c, mask_c)", "masked_recon_loss_c(x_hat_c, x_c, mask_c)")
        
    elif is_dl:
        # Inject after compute_mae
        content = re.sub(
            r'(def compute_mae\(imputed, truth, mask\):\n.*?return np\.mean\(np\.abs\(diff\)\) if obs\.sum\(\) > 0 else 0\.0\n)',
            r'\1\n' + injection_code + '\n',
            content,
            flags=re.DOTALL
        )
        # Replace inline recon_loss for StandardVAE
        content = content.replace("recon_loss = torch.sum(((x_hat - x_mb) * M_obs)**2) / (torch.sum(M_obs) + 1e-8)", "recon_loss = masked_recon_loss_c(x_hat, x_mb, m_mb)")
        
    else:
        # MNARSVAE and MYEVAE
        content = re.sub(
            r'(def masked_mse\(preds, targets, mask\):\n.*?return loss\n)',
            r'\1\n' + injection_code + '\n',
            content,
            flags=re.DOTALL
        )
        content = content.replace("masked_mse(x_hat_c, xc_mb, mc_mb)", "masked_recon_loss_c(x_hat_c, xc_mb, mc_mb)")
        
    with open(filepath, 'w') as f:
        f.write(content)

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"
patch_file(f"{base_dir}/train_sigma_optuna.py", is_sigma=True)
patch_file(f"{base_dir}/train_mnarsvae_optuna.py")
patch_file(f"{base_dir}/train_myevae_optuna.py")
patch_file(f"{base_dir}/train_dl_baselines_optuna.py", is_dl=True)

print("Patching complete.")
