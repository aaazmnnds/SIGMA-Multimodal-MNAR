import re
with open('train_mnarsvae_optuna.py', 'r') as f:
    code = f.read()

optuna_tune_0 = """
    print(f"[MNARSVAE-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    def make_loader_0(idx, shuffle):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X_c_masked[idx]),
            torch.FloatTensor(mask_c_full[idx]),
            torch.FloatTensor(T_all[idx]),
            torch.FloatTensor(E_all[idx])
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

    for k, fold_dict in enumerate(folds, 1):"""

pattern = re.compile(r"    all_rmse_c = \[\]\s*all_mae_c = \[\]\s*for k, fold_dict in enumerate\(folds, 1\):.*?print\(f\"\[MNARSVAE-OPTUNA\] Fold \{k\}/5 \| Retraining with best params\.\.\.\"\)", re.DOTALL)
new_code = pattern.sub(optuna_tune_0, code)

with open('train_mnarsvae_optuna.py', 'w') as f:
    f.write(new_code)
