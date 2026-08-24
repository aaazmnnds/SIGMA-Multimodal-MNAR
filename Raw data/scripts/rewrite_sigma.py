import re
with open('train_sigma_optuna.py', 'r') as f:
    code = f.read()

# Define the new Optuna fold 0 code
optuna_tune_0 = """
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

    for k, fold_dict in enumerate(folds, 1):"""

# Regex substitution to replace from the start of the enumerate loop up to the model instantiation
pattern = re.compile(r"    optuna\.logging\.set_verbosity\(optuna\.logging\.WARNING\).*?print\(f\"\[SIGMA-OPTUNA\] Fold \{k\}/5 \| Retraining with best params\.\.\.\"\)\s*np\.random\.seed\(42\)\s*torch\.manual_seed\(42\)", re.DOTALL)
new_code = pattern.sub(optuna_tune_0, code)

# Remove the consolidated json saving logic at the end
consolidated_pattern = re.compile(r"    # Save best hyperparameters from all folds.*?if all_params:\s*pd\.DataFrame\(all_params\)\.to_csv\(f\"{out_path}/all_folds_best_params\.csv\", index=False\)", re.DOTALL)
new_code = consolidated_pattern.sub("", new_code)

with open('train_sigma_optuna.py', 'w') as f:
    f.write(new_code)
