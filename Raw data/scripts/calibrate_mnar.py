# MNAR calibration script for SIGMA study. Calibrates alpha and beta parameters to achieve target missingness proportions under threshold-interaction MNAR.

import os
import numpy as np
import pandas as pd
from scipy.special import expit

BASE = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data"
PROCESSED_DIR = f"{BASE}/processed"
CLINICAL_FILE = f"{PROCESSED_DIR}/metabric_clinical.csv"
GENOMIC_FILE = f"{PROCESSED_DIR}/metabric_genomic.csv"
OUTPUT_DIR = f"{BASE}/mnar_simulation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

alpha_grid = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
beta_grid = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
target_rates = [0.20, 0.40, 0.60]
severity_labels = ['light', 'moderate', 'severe']
n_seeds = 10

clinical_df = pd.read_csv(CLINICAL_FILE)
genomic_df = pd.read_csv(GENOMIC_FILE)

age_array = clinical_df['AGE_AT_DIAGNOSIS'].values
tau = np.nanmedian(age_array)

print(f"Median age used as threshold tau: {tau}")

age_indicator = (age_array > tau).astype(float)

def calibrate(modality_name, numeric_features, target_rate, severity):
    best_alpha = None
    best_beta = None
    best_diff = float('inf')
    best_achieved = None
    
    for alpha in alpha_grid:
        for beta in beta_grid:
            achieved_rates = []
            for seed in range(n_seeds):
                np.random.seed(seed)
                prob = expit(alpha * age_indicator[:, None] * np.abs(numeric_features) + beta)
                mask = np.random.binomial(1, prob)
                achieved_rates.append(mask.mean())
                
            avg_rate = np.mean(achieved_rates)
            diff = abs(avg_rate - target_rate)
            
            print(f"[CALIBRATION] Modality: {modality_name} | Severity: {severity} | Testing alpha={alpha}, beta={beta} | Achieved rate: {avg_rate}")
            
            if diff < best_diff:
                best_diff = diff
                best_alpha = alpha
                best_beta = beta
                best_achieved = avg_rate
                
    return best_alpha, best_beta, best_achieved, best_diff

results = []
final_masks = {}

# Prepare feature matrices
clinical_features = clinical_df.select_dtypes(include=[np.number]).drop(columns=['AGE_AT_DIAGNOSIS', 'PATIENT_ID'], errors='ignore').values

genomic_num_cols = genomic_df.select_dtypes(include=[np.number]).columns
genomic_expr_cols = [c for c in genomic_num_cols if not c.endswith('_CNA') and c != 'PATIENT_ID']
genomic_features = genomic_df[genomic_expr_cols].values

for modality_name, numeric_features in [("clinical", clinical_features), ("genomic", genomic_features)]:
    masks = []
    
    for target_rate, severity in zip(target_rates, severity_labels):
        alpha, beta, achieved, diff = calibrate(modality_name, numeric_features, target_rate, severity)
        
        results.append({
            'modality': modality_name,
            'severity': severity,
            'target_rate': target_rate,
            'alpha': alpha,
            'beta': beta,
            'achieved_rate': achieved,
            'abs_error': diff
        })
        
        np.random.seed(42)
        prob = expit(alpha * age_indicator[:, None] * np.abs(numeric_features) + beta)
        mask = np.random.binomial(1, prob)
        masks.append(mask)
        
    final_masks[modality_name] = np.array(masks)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/calibration_results.csv", index=False)

print("\nCalibration Summary Table:")
print(results_df.to_string(index=False))

np.save(f"{OUTPUT_DIR}/mnar_masks_metabric_clinical.npy", final_masks['clinical'])
np.save(f"{OUTPUT_DIR}/mnar_masks_metabric_genomic.npy", final_masks['genomic'])
