# Preprocessing script for SIGMA study. METABRIC filtered to Nature 2012 cohort with complete omics and valid survival data. TCGA used for genomic-only external validation.

import os
import pandas as pd
import numpy as np

BASE = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data"
METABRIC_DIR = f"{BASE}/original_datasets/brca_metabric"
TCGA_DIR = f"{BASE}/original_datasets/brca_tcga_pan_can_atlas_2018"
OUTPUT_DIR = f"{BASE}/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full PAM50 list
pam50_genes = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", 
    "CDC6", "CDH3", "CEACAM5", "CKS2", "CTSL2", "CXXC5", "EGFR", "ERBB2", "ESR1", 
    "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", 
    "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", 
    "NAT1", "NDC80", "NUF2", "ORC6", "PCNA", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", 
    "SLC39A6", "TMEM45B", "UBE2C"
]

print("Starting METABRIC Preprocessing Pipeline...")

# Step 1
print("[METABRIC] Step 1: Load cohort filter")
with open(f"{METABRIC_DIR}/case_lists/cases_nature_2012.txt", "r") as f:
    nature2012_ids = set()
    for line in f:
        if line.startswith("case_list_ids:"):
            parts = line.strip().split("\t")
            first_id = parts[0].replace("case_list_ids:", "").strip()
            if first_id:
                nature2012_ids.add(first_id)
            for p in parts[1:]:
                if p.strip():
                    nature2012_ids.add(p.strip())
print(f"Loaded {len(nature2012_ids)} patient IDs for Nature 2012 cohort.")

# Step 2
print("[METABRIC] Step 2: Load clinical data")
metabric_patient = pd.read_csv(f"{METABRIC_DIR}/data_clinical_patient.txt", sep="\t", comment="#")
cols_patient = ['PATIENT_ID', 'AGE_AT_DIAGNOSIS', 'CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY', 'OS_MONTHS', 'OS_STATUS']
metabric_patient = metabric_patient[cols_patient]

metabric_sample = pd.read_csv(f"{METABRIC_DIR}/data_clinical_sample.txt", sep="\t", comment="#")
cols_sample = ['PATIENT_ID', 'ER_STATUS', 'HER2_STATUS', 'PR_STATUS']
metabric_sample = metabric_sample[cols_sample]

metabric_clin = pd.merge(metabric_patient, metabric_sample, on='PATIENT_ID', how='inner')
print(f"Loaded clinical data. Patients after merge: {len(metabric_clin)}")

# Step 3
print("[METABRIC] Step 3: Load expression data for PAM50 genes (includes MKI67 and EGFR)")
expr_file = f"{METABRIC_DIR}/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
expr_df = pd.read_csv(expr_file, sep="\t")
pam50_expr = expr_df[expr_df['Hugo_Symbol'].isin(pam50_genes)].copy()

# For each duplicate Hugo_Symbol, keep the row with highest mean across patient columns
patient_cols = [c for c in pam50_expr.columns if c not in ['Hugo_Symbol', 'Entrez_Gene_Id']]
pam50_expr['_mean'] = pam50_expr[patient_cols].mean(axis=1)
pam50_expr = pam50_expr.sort_values('_mean', ascending=False).drop_duplicates(subset=['Hugo_Symbol']).drop(columns=['_mean'])

missing_genes = set(pam50_genes) - set(pam50_expr['Hugo_Symbol'])
if missing_genes:
    print(f"WARNING: Missing PAM50 genes in METABRIC expression data: {missing_genes}")

pam50_expr = pam50_expr.drop(columns=['Entrez_Gene_Id'], errors='ignore')
pam50_expr = pam50_expr.set_index('Hugo_Symbol').transpose().reset_index().rename(columns={'index': 'PATIENT_ID'})
metabric_clin = pd.merge(metabric_clin, pam50_expr, on='PATIENT_ID', how='inner')
print(f"Merged PAM50 expression. Patients after merge: {len(metabric_clin)}")

# Step 4
print("[METABRIC] Step 4: Apply cohort filter")
n_before = len(metabric_clin)
metabric_clin = metabric_clin[metabric_clin['PATIENT_ID'].isin(nature2012_ids)]
print(f"Dropped {n_before - len(metabric_clin)} patients not in Nature 2012 cohort.")

# Step 5
print("[METABRIC] Step 5: Apply survival quality filter")
n_before = len(metabric_clin)
metabric_clin['OS_MONTHS'] = pd.to_numeric(metabric_clin['OS_MONTHS'], errors='coerce')
metabric_clin = metabric_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
metabric_clin = metabric_clin[metabric_clin['OS_MONTHS'] > 0]
metabric_clin = metabric_clin[metabric_clin['OS_STATUS'] != '']
metabric_clin['OS_STATUS'] = metabric_clin['OS_STATUS'].apply(lambda x: 1 if '1:' in str(x) or 'DECEASED' in str(x).upper() else 0)
metabric_clin = metabric_clin.rename(columns={'OS_MONTHS': 'T', 'OS_STATUS': 'E'})
print(f"Dropped {n_before - len(metabric_clin)} patients due to invalid survival data.")

# Step 6
print("[METABRIC] Step 6: Encode clinical categorical variables")
for col in ['CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY', 'ER_STATUS', 'HER2_STATUS', 'PR_STATUS']:
    print(f"Unique values in {col} before encoding: {metabric_clin[col].unique()}")

for col in ['CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY']:
    metabric_clin[col] = metabric_clin[col].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else (0 if str(x).strip().upper() == 'NO' else np.nan))

for col in ['ER_STATUS', 'HER2_STATUS', 'PR_STATUS']:
    metabric_clin[col] = metabric_clin[col].apply(lambda x: 1 if str(x).strip().upper() == 'POSITIVE' else (0 if str(x).strip().upper() == 'NEGATIVE' else np.nan))

n_before = len(metabric_clin)
metabric_clin = metabric_clin.dropna(subset=['CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY', 'ER_STATUS', 'HER2_STATUS', 'PR_STATUS'])
print(f"Dropped {n_before - len(metabric_clin)} patients with missing categorical clinical features.")

# Step 8 (Former separate load removed as PAM50 expression is now merged in Step 3)
print("[METABRIC] Step 8: PAM50 expression features were loaded and merged in Step 3")

# Step 9
print("[METABRIC] Step 9: Load CNA features")
cna_file = f"{METABRIC_DIR}/data_cna.txt"
cna_df = pd.read_csv(cna_file, sep="\t")
pam50_cna = cna_df[cna_df['Hugo_Symbol'].isin(pam50_genes)].copy()

# For each duplicate Hugo_Symbol, keep the row with highest mean across patient columns
patient_cols = [c for c in pam50_cna.columns if c not in ['Hugo_Symbol', 'Entrez_Gene_Id']]
pam50_cna['_mean'] = pam50_cna[patient_cols].mean(axis=1)
pam50_cna = pam50_cna.sort_values('_mean', ascending=False).drop_duplicates(subset=['Hugo_Symbol']).drop(columns=['_mean'])

pam50_cna = pam50_cna.drop(columns=['Entrez_Gene_Id'], errors='ignore')
pam50_cna = pam50_cna.set_index('Hugo_Symbol').transpose()
pam50_cna.columns = [str(c) + '_CNA' for c in pam50_cna.columns]
pam50_cna = pam50_cna.reset_index().rename(columns={'index': 'PATIENT_ID'})
metabric_clin = pd.merge(metabric_clin, pam50_cna, on='PATIENT_ID', how='inner')

# Step 7 (Moved after all merges)
print("[METABRIC] Step 7: Standardize continuous clinical features")
for col in ['AGE_AT_DIAGNOSIS', 'MKI67', 'EGFR']:
    metabric_clin[col] = pd.to_numeric(metabric_clin[col])
    metabric_clin[col] = (metabric_clin[col] - metabric_clin[col].mean()) / metabric_clin[col].std()

# Step 10
print("[METABRIC] Step 10: Final intersection")
print(f"Final METABRIC N: {len(metabric_clin)}")

# Step 11
print("[METABRIC] Step 11: Split into clinical and genomic feature matrices and save")
metabric_clinical = metabric_clin[['PATIENT_ID', 'AGE_AT_DIAGNOSIS', 'CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY', 'ER_STATUS', 'HER2_STATUS', 'PR_STATUS', 'MKI67', 'EGFR']]
cna_cols = [c for c in metabric_clin.columns if c.endswith('_CNA')]
expr_cols = [c for c in metabric_clin.columns if c in pam50_genes]
metabric_genomic = metabric_clin[['PATIENT_ID'] + expr_cols + cna_cols]
metabric_survival = metabric_clin[['PATIENT_ID', 'T', 'E']]

metabric_clinical.to_csv(f"{OUTPUT_DIR}/metabric_clinical.csv", index=False)
metabric_genomic.to_csv(f"{OUTPUT_DIR}/metabric_genomic.csv", index=False)
metabric_survival.to_csv(f"{OUTPUT_DIR}/metabric_survival.csv", index=False)

print(f"metabric_clinical.csv shape: {metabric_clinical.shape}")
print(f"metabric_genomic.csv shape: {metabric_genomic.shape}")
print(f"metabric_survival.csv shape: {metabric_survival.shape}")


print("\nStarting TCGA Preprocessing Pipeline...")

# Step 1
print("[TCGA] Step 1: Load clinical data")
tcga_patient = pd.read_csv(f"{TCGA_DIR}/data_clinical_patient.txt", sep="\t", comment="#")
cols_tcga = ['PATIENT_ID', 'AGE', 'SUBTYPE', 'OS_MONTHS', 'OS_STATUS']
tcga_clin = tcga_patient[cols_tcga].copy()

# Step 2
print("[TCGA] Step 2: Apply survival quality filter")
n_before = len(tcga_clin)
tcga_clin['OS_MONTHS'] = pd.to_numeric(tcga_clin['OS_MONTHS'], errors='coerce')
tcga_clin = tcga_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
tcga_clin = tcga_clin[tcga_clin['OS_MONTHS'] > 0]
tcga_clin = tcga_clin[tcga_clin['OS_STATUS'] != '']
tcga_clin['OS_STATUS'] = tcga_clin['OS_STATUS'].apply(lambda x: 1 if '1:' in str(x) or 'DECEASED' in str(x).upper() else 0)
tcga_clin = tcga_clin.rename(columns={'OS_MONTHS': 'T', 'OS_STATUS': 'E'})
print(f"Dropped {n_before - len(tcga_clin)} patients due to invalid survival data.")

# Step 3
print("[TCGA] Step 3: Derive ER_STATUS proxy from SUBTYPE")
def map_subtype(subtype):
    subtype = str(subtype).strip()
    if subtype in ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Normal']:
        return 1
    elif subtype in ['BRCA_Her2', 'BRCA_Basal']:
        return 0
    return np.nan

tcga_clin['ER_STATUS_PROXY'] = tcga_clin['SUBTYPE'].apply(map_subtype)

# Step 4
print("[TCGA] Step 4: Standardize AGE")
tcga_clin['AGE'] = pd.to_numeric(tcga_clin['AGE'])
tcga_clin['AGE'] = (tcga_clin['AGE'] - tcga_clin['AGE'].mean()) / tcga_clin['AGE'].std()

# Step 5
print("[TCGA] Step 5: Load PAM50 expression features")
tcga_expr_file = f"{TCGA_DIR}/data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt"
tcga_expr_df = pd.read_csv(tcga_expr_file, sep="\t")
tcga_pam50_expr = tcga_expr_df[tcga_expr_df['Hugo_Symbol'].isin(pam50_genes)].copy()

# For each duplicate Hugo_Symbol, keep the row with highest mean across patient columns
patient_cols = [c for c in tcga_pam50_expr.columns if c not in ['Hugo_Symbol', 'Entrez_Gene_Id']]
tcga_pam50_expr['_mean'] = tcga_pam50_expr[patient_cols].mean(axis=1)
tcga_pam50_expr = tcga_pam50_expr.sort_values('_mean', ascending=False).drop_duplicates(subset=['Hugo_Symbol']).drop(columns=['_mean'])

missing_tcga_genes = set(pam50_genes) - set(tcga_pam50_expr['Hugo_Symbol'])
if missing_tcga_genes:
    print(f"WARNING: Missing PAM50 genes in TCGA expression data: {missing_tcga_genes}")

tcga_pam50_expr = tcga_pam50_expr.drop(columns=['Entrez_Gene_Id'], errors='ignore')
tcga_pam50_expr = tcga_pam50_expr.set_index('Hugo_Symbol').transpose().reset_index().rename(columns={'index': 'PATIENT_ID'})
tcga_pam50_expr['PATIENT_ID'] = tcga_pam50_expr['PATIENT_ID'].astype(str).str[:12]
tcga_pam50_expr = tcga_pam50_expr.drop_duplicates(subset=['PATIENT_ID'])
tcga_clin = pd.merge(tcga_clin, tcga_pam50_expr, on='PATIENT_ID', how='inner')

# Step 6
print("[TCGA] Step 6: Load CNA features")
tcga_cna_file = f"{TCGA_DIR}/data_cna.txt"
tcga_cna_df = pd.read_csv(tcga_cna_file, sep="\t")
tcga_pam50_cna = tcga_cna_df[tcga_cna_df['Hugo_Symbol'].isin(pam50_genes)].copy()

# For each duplicate Hugo_Symbol, keep the row with highest mean across patient columns
patient_cols = [c for c in tcga_pam50_cna.columns if c not in ['Hugo_Symbol', 'Entrez_Gene_Id']]
tcga_pam50_cna['_mean'] = tcga_pam50_cna[patient_cols].mean(axis=1)
tcga_pam50_cna = tcga_pam50_cna.sort_values('_mean', ascending=False).drop_duplicates(subset=['Hugo_Symbol']).drop(columns=['_mean'])

tcga_pam50_cna = tcga_pam50_cna.drop(columns=['Entrez_Gene_Id'], errors='ignore')
tcga_pam50_cna = tcga_pam50_cna.set_index('Hugo_Symbol').transpose()
tcga_pam50_cna.columns = [str(c) + '_CNA' for c in tcga_pam50_cna.columns]
tcga_pam50_cna = tcga_pam50_cna.reset_index().rename(columns={'index': 'PATIENT_ID'})
tcga_pam50_cna['PATIENT_ID'] = tcga_pam50_cna['PATIENT_ID'].astype(str).str[:12]
tcga_pam50_cna = tcga_pam50_cna.drop_duplicates(subset=['PATIENT_ID'])
tcga_clin = pd.merge(tcga_clin, tcga_pam50_cna, on='PATIENT_ID', how='inner')

# Step 7
print("[TCGA] Step 7: Final intersection")
print(f"Final TCGA N: {len(tcga_clin)}")

# Step 8
print("[TCGA] Step 8: Save output files to OUTPUT_DIR")
tcga_expr_cols = [c for c in tcga_clin.columns if c in pam50_genes]
tcga_cna_cols = [c for c in tcga_clin.columns if c.endswith('_CNA')]

tcga_genomic = tcga_clin[['PATIENT_ID'] + tcga_expr_cols + tcga_cna_cols]
tcga_survival = tcga_clin[['PATIENT_ID', 'T', 'E']]
tcga_subtype = tcga_clin[['PATIENT_ID', 'SUBTYPE', 'ER_STATUS_PROXY']]

tcga_genomic.to_csv(f"{OUTPUT_DIR}/tcga_genomic.csv", index=False)
tcga_survival.to_csv(f"{OUTPUT_DIR}/tcga_survival.csv", index=False)
tcga_subtype.to_csv(f"{OUTPUT_DIR}/tcga_subtype.csv", index=False)

print(f"tcga_genomic.csv shape: {tcga_genomic.shape}")
print(f"tcga_survival.csv shape: {tcga_survival.shape}")
print(f"tcga_subtype.csv shape: {tcga_subtype.shape}")

print("\n=== SUMMARY ===")
print(f"METABRIC Final N: {len(metabric_survival)}")
print(f"METABRIC clinical features: {metabric_clinical.shape[1] - 1}") # Exclude PATIENT_ID
print(f"METABRIC expression features: {len(expr_cols)}")
print(f"METABRIC CNA features: {len(cna_cols)}")

print(f"TCGA Final N: {len(tcga_survival)}")
print(f"TCGA clinical features: N/A (genomic validation only)")
print(f"TCGA expression features: {len(tcga_expr_cols)}")
print(f"TCGA CNA features: {len(tcga_cna_cols)}")
