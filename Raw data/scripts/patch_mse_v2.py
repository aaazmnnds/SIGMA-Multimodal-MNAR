import re
import os

injection_code = """
BINARY_COLS_C = [1, 2, 3, 4, 5, 6]
CONTINUOUS_COLS_C = [0, 7, 8]

def masked_recon_loss_c(preds, targets, mask):
    \"\"\"Mixed MSE + BCE reconstruction loss for clinical features.\"\"\"
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
"""

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Add import torch.nn.functional as F if not exists
    if "import torch.nn.functional as F" not in content:
        content = re.sub(r'(import torch\n)', r'\1import torch.nn.functional as F\n', content)

    # Replace old masked_recon_loss_c with new one
    content = re.sub(
        r'BINARY_COLS_C = \[1, 2, 3, 4, 5, 6\].*?return loss\n',
        injection_code.strip() + '\n',
        content,
        flags=re.DOTALL
    )

    with open(filepath, 'w') as f:
        f.write(content)

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"
patch_file(f"{base_dir}/train_sigma_optuna.py")
patch_file(f"{base_dir}/train_mnarsvae_optuna.py")
patch_file(f"{base_dir}/train_myevae_optuna.py")
patch_file(f"{base_dir}/train_dl_baselines_optuna.py")

print("Patching complete.")
