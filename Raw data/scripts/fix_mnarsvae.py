import re
with open('train_mnarsvae_optuna.py', 'r') as f:
    code = f.read()

loop_header = """    for k, fold_dict in enumerate(folds, 1):
        train_idx = fold_dict['train']
        val_idx = fold_dict['val']
        test_idx = fold_dict['test']
        
        def make_loader(idx, shuffle):
            from torch.utils.data import TensorDataset, DataLoader
            return DataLoader(TensorDataset(
                torch.FloatTensor(X_c_masked[idx]),
                torch.FloatTensor(mask_c_full[idx]),
                torch.FloatTensor(T_all[idx]),
                torch.FloatTensor(E_all[idx])
            ), batch_size=args.batch_size, shuffle=shuffle)

        train_loader = make_loader(train_idx, True)
        val_loader = make_loader(val_idx, False)
        test_loader = make_loader(test_idx, False)
        
        model = MNARSVAEModel"""

pattern = re.compile(r"    for k, fold_dict in enumerate\(folds, 1\):\s*model = MNARSVAEModel", re.DOTALL)
new_code = pattern.sub(loop_header, code)

with open('train_mnarsvae_optuna.py', 'w') as f:
    f.write(new_code)
