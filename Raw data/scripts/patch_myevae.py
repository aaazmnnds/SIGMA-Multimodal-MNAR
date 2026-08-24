import re

with open("/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts/train_myevae_optuna.py", 'r') as f:
    content = f.read()

# We need to insert the fold 0 tuning logic before the `for k, fold_dict in enumerate(folds, 1):` loop.

insert_code = """
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
"""

# Find where the `results = []` line is and insert right after it
start_idx = content.find("results = []")
end_idx = content.find("for k, fold_dict in enumerate(folds, 1):")

# Remove the per-fold tuning inside the loop
# Specifically: lines from `print(f"[MYEVAE-OPTUNA] Fold {k}/5 | Running` up to `print(f"[MYEVAE-OPTUNA] Fold {k}/5 | Retraining with best params...")`

remove_regex = r'        print\(f"\[MYEVAE-OPTUNA\] Fold \{k\}/5 \| Running \{args\.n_trials\} Optuna trials\.\.\."\).*?with open\(f"\{out_path\}/fold_\{k\}_best_params\.json", \'w\'\) as f:\n            json\.dump\(best_params, f, indent=4\)\n        \n        print\(f"\[MYEVAE-OPTUNA\] Fold \{k\}/5 \| Retraining with best params\.\.\."\)'

content_clean = re.sub(remove_regex, r'', content, flags=re.DOTALL)
content_clean = content_clean.replace("optuna.logging.set_verbosity(optuna.logging.WARNING)", "")

final_content = content_clean[:start_idx] + "results = []\n" + insert_code + "\n" + content_clean[end_idx:]

with open("/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts/train_myevae_optuna.py", 'w') as f:
    f.write(final_content)
