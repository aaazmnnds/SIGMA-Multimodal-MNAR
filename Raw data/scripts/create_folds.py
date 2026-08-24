# Fold generation script for SIGMA study. Creates stratified 5-fold CV splits with train/val/test structure. All downstream models must use these splits for fair comparison.

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data"
PROCESSED_DIR = f"{BASE}/processed"
SURVIVAL_FILE = f"{PROCESSED_DIR}/metabric_survival.csv"
OUTPUT_DIR = f"{PROCESSED_DIR}/folds"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(SURVIVAL_FILE)
E = df['E'].values
indices = np.arange(len(df))

full_event_rate = E.mean()
print(f"Full dataset event rate: {full_event_rate:.4f} (N={len(df)})")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []
summary_data = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, E), 1):
    print(f"[FOLDS] Creating fold {fold_idx} of 5")
    
    E_train_val = E[train_val_idx]
    
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=0.1, 
        stratify=E_train_val, 
        random_state=42
    )
    
    folds.append({
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    })
    
    summary_data.append({
        'fold': fold_idx,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'train_event_rate': E[train_idx].mean(),
        'val_event_rate': E[val_idx].mean(),
        'test_event_rate': E[test_idx].mean()
    })

pkl_path = f"{OUTPUT_DIR}/metabric_folds.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(folds, f)

csv_path = f"{OUTPUT_DIR}/fold_summary.csv"
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(csv_path, index=False)

print("\nFold Summary Table:")
print(summary_df.to_string(index=False))

print(f"\nOutputs saved:")
print(f"1. {pkl_path}")
print(f"2. {csv_path}")
