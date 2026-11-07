"""
benchmark_timing.py

Benchmarks execution time for SIGMA (EF, CMA, PoE) models:
- Single epoch training time (forward + backward + optimizer step across batches, mean over 5 epochs)
- Single forward inference pass (encoder + decoder, torch.no_grad) on full N=1979 in one batch (mean over 20 repeats)
- Optuna trial simulation (50 epochs estimate = single epoch time * 50)

Outputs clean ASCII table to stdout.
"""

import sys
import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

# Ensure scripts directory is on sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Import model and components directly from train_sigma_optuna
import train_sigma_optuna
from train_sigma_optuna import (
    SIGMA,
    masked_recon_loss_c,
    masked_mse,
    kl_divergence,
    cox_partial_likelihood,
    infonce_loss,
)


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Exact METABRIC cohort dimensions
    N = 1979
    p_c = 9
    p_g = 1000
    batch_size = 128
    d = 64  # Representative latent dimension from hyperparameter tuning

    # Generate synthetic tensors matching dimensions
    torch.manual_seed(42)
    x_c = torch.randn(N, p_c)
    mask_c = torch.randint(0, 2, (N, p_c)).float()
    mask_c[:, 0] = 0.0  # AGE feature is always fully observed in METABRIC

    x_g = torch.randn(N, p_g)
    mask_g = torch.zeros(N, p_g)
    mask_g[:, :49] = torch.randint(0, 2, (N, 49)).float()  # Expression features masked

    # Survival targets
    T = torch.rand(N) * 150.0 + 1.0  # Follow-up times
    E = torch.randint(0, 2, (N,)).float()  # Event indicator

    # DataLoader for training
    dataset = TensorDataset(x_c, mask_c, x_g, mask_g, T, E)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Full cohort tensors for inference
    full_x_c = x_c.to(device)
    full_mask_c = mask_c.to(device)
    full_x_g = x_g.to(device)
    full_mask_g = mask_g.to(device)

    fusion_strategies = ['EF', 'CMA', 'PoE']
    results = {}

    for fusion in fusion_strategies:
        model = SIGMA(p_c, p_g, d, fusion).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # -------------------------------------------------------------
        # 1. Single Epoch Training Time (mean over 5 repeats)
        # -------------------------------------------------------------
        # Warmup (1 epoch)
        model.train()
        for bx_c, bm_c, bx_g, bm_g, bT, bE in train_loader:
            bx_c, bm_c = bx_c.to(device), bm_c.to(device)
            bx_g, bm_g = bx_g.to(device), bm_g.to(device)
            bT, bE = bT.to(device), bE.to(device)

            optimizer.zero_grad()
            x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(bx_c, bm_c, bx_g, bm_g)
            l_c = masked_recon_loss_c(x_hat_c, bx_c, bm_c)
            l_g = masked_mse(x_hat_g, bx_g, bm_g)
            l_kl = kl_divergence(mu_shared, log_var_shared)
            l_cox = cox_partial_likelihood(eta, bT, bE)
            l_cross = infonce_loss(z_c, z_g, 0.1)
            loss = l_c + l_g + 0.1 * l_kl + 1.0 * l_cox + 0.5 * l_cross
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        n_train_repeats = 5
        train_times = []

        for _ in range(n_train_repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            model.train()
            for bx_c, bm_c, bx_g, bm_g, bT, bE in train_loader:
                bx_c, bm_c = bx_c.to(device), bm_c.to(device)
                bx_g, bm_g = bx_g.to(device), bm_g.to(device)
                bT, bE = bT.to(device), bE.to(device)

                optimizer.zero_grad()
                x_hat_c, x_hat_g, eta, z_c, z_g, z_shared, mu_shared, log_var_shared = model(bx_c, bm_c, bx_g, bm_g)
                l_c = masked_recon_loss_c(x_hat_c, bx_c, bm_c)
                l_g = masked_mse(x_hat_g, bx_g, bm_g)
                l_kl = kl_divergence(mu_shared, log_var_shared)
                l_cox = cox_partial_likelihood(eta, bT, bE)
                l_cross = infonce_loss(z_c, z_g, 0.1)
                loss = l_c + l_g + 0.1 * l_kl + 1.0 * l_cox + 0.5 * l_cross
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            train_times.append(t_end - t_start)

        mean_epoch_time = sum(train_times) / len(train_times)

        # -------------------------------------------------------------
        # 2. Inference Time: full N=1979 in one batch (mean over 20 repeats)
        # -------------------------------------------------------------
        model.eval()
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(full_x_c, full_mask_c, full_x_g, full_mask_g)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        n_inf_repeats = 20
        inf_times = []

        for _ in range(n_inf_repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                _ = model(full_x_c, full_mask_c, full_x_g, full_mask_g)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            inf_times.append(t_end - t_start)

        mean_inf_time = sum(inf_times) / len(inf_times)

        # -------------------------------------------------------------
        # 3. Optuna Trial Simulation (50 epochs estimate)
        # -------------------------------------------------------------
        optuna_trial_time = mean_epoch_time * 50.0

        results[fusion] = {
            'epoch_time_s': mean_epoch_time,
            'inference_time_ms': mean_inf_time * 1000.0,
            'optuna_trial_s': optuna_trial_time,
            'optuna_trial_min': optuna_trial_time / 60.0,
        }

    # Print clean summary table
    print("\n" + "=" * 80)
    print("SIGMA COMPUTATIONAL TIMING BENCHMARK REPORT")
    print(f"Cohort dimensions: N={N}, p_c={p_c}, p_g={p_g}, batch_size={batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    header = f"{'Strategy':<12} | {'Epoch Time (s)':<16} | {'Inference N=1979 (ms)':<22} | {'Optuna Trial (50 ep)':<20}"
    print(header)
    print("-" * 80)
    for fusion in fusion_strategies:
        r = results[fusion]
        row = f"{fusion:<12} | {r['epoch_time_s']:>14.3f} s | {r['inference_time_ms']:>20.2f} ms | {r['optuna_trial_s']:>10.2f} s ({r['optuna_trial_min']:.2f} m)"
        print(row)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    run_benchmark()
