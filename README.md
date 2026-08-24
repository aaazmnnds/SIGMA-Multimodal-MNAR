# SIGMA: Survival Integration via Generative Multi-omics Alignment

PyTorch implementation of **SIGMA**, a multimodal variational autoencoder designed to address **Missing Not At Random (MNAR)** mechanisms in heterogeneous clinical and genomic datasets within a survival-aware generative framework.

---

## 📌 Key Architectural Features

- **Modality-Specific Encoders:** Maps clinical ($P_c = 9$) and genomic ($P_g = 98$) inputs to unimodal Gaussian latent spaces ($\mathbf{z}_c, \mathbf{z}_g$) with explicit mask concatenation $[\mathbf{X}, \mathbf{M}]$.
- **Latent Alignment:** InfoNCE-based contrastive loss ($\mathcal{L}_{\text{cross}}$) enforcing patient-level correspondence across modalities.
- **Multimodal Fusion:** Evaluates three distinct fusion strategies:
  - **Early Fusion (EF):** Linear projection of concatenated unimodal latents.
  - **Cross-Modal Attention (CMA):** Bidirectional cross-attention exchange before projection.
  - **Product of Experts (PoE):** Analytic Gaussian product of unimodal posteriors.
- **Survival Guidance:** Internal Cox proportional hazards network optimizing $\mathcal{L}_{\text{Cox}}$ on the shared latent representation $\mathbf{z}_{\text{shared}}$.

---

## 📂 Repository Structure & Script Directory

All computational experiments and evaluation protocols are located under `Raw data/scripts/`:

```text
Raw data/scripts/
├── sigma_model.py              # Core SIGMA model architecture, encoders, decoders, fusion modules (EF, CMA, PoE), and loss functions
├── preprocess_sigma.py         # METABRIC and TCGA clinical & genomic data extraction, cleaning, and z-score normalization
├── calibrate_mnar.py           # Threshold-interaction MNAR simulation grid search for targeted missingness rates
├── create_folds.py             # 5-fold cross-validation split generation preserving overall survival distributions
│
├── Training & Hyperparameter Tuning (Optuna)
│   ├── train_sigma_optuna.py       # Full SIGMA models (EF, CMA, PoE) with Optuna TPE tuning (50 trials on fold 0)
│   ├── train_sigma_ablation.py     # 5 ablation variants (w/o L_cross, w/o L_Cox, w/o Mask Input, Clinical-only, Genomic-only)
│   ├── train_dl_baselines_optuna.py# Deep learning baselines (StandardVAE, MNAR-SVAE, MyeVAE) with Optuna TPE
│   ├── train_baselines_optuna.py   # Classical baseline runners (MICE, missForest) and iterative methods
│   ├── train_gain_mida.py          # GAIN and MIDA baseline training and evaluation runners
│   └── train_mnarsvae_optuna.py    # Unimodal MNAR-SVAE baseline training runner
│
├── Downstream Evaluation & Benchmarking
│   ├── evaluate_survival.py        # Downstream survival models (Random Survival Forests, XGBoost, DeepSurv, DeepHit)
│   ├── evaluate_pam50.py           # PAM50 molecular subtype classification on reconstructed genomic features
│   ├── evaluate_ablation.py        # RSF survival C-index and PAM50 classification evaluation for the 5 ablation variants
│   ├── evaluate_integration.py     # Canonical Correlation Analysis (CCA) and t-SNE Silhouette cluster score evaluation
│   ├── evaluate_crossmodal.py      # Cross-modal reconstruction evaluation (Clinical-to-Genomic and Genomic-to-Clinical)
│   └── evaluate_direct_survival.py # Direct internal Cox hazard survival evaluation from latent representations
│
└── Visualization
    └── generate_figures.py         # Generates figures for imputation RMSE, C-index, PAM50 accuracy, and t-SNE projections
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/aaazmnnds/SIGMA-Multimodal-MNAR.git
cd SIGMA-Multimodal-MNAR
pip install -r requirements.txt
```

### 2. Dataset Acquisition

The METABRIC and TCGA cohorts are publicly available via cBioPortal:
- **METABRIC Breast Cancer Cohort (`brca_metabric`):** [cBioPortal Study](https://www.cbioportal.org/study/summary?id=brca_metabric) / [DataHub](https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric)
- **TCGA PanCancer Atlas BRCA (`brca_tcga_pan_can_atlas_2018`):** [cBioPortal Study](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018) / [DataHub](https://github.com/cBioPortal/datahub/tree/master/public/brca_tcga_pan_can_atlas_2018)

Place raw dataset files in `Raw data/original_datasets/brca_metabric` and `Raw data/original_datasets/brca_tcga_pan_can_atlas_2018`.

---

## 🧪 Reproducing Experiments

### Step 1: Preprocessing & Fold Generation
```bash
cd "Raw data/scripts"
python3 preprocess_sigma.py
python3 create_folds.py
```

### Step 2: MNAR Calibration
```bash
python3 calibrate_mnar.py
```

### Step 3: Model Training & Hyperparameter Tuning (Optuna)
```bash
# 1. Train Full SIGMA Fusion Models (EF, CMA, PoE)
python3 train_sigma_optuna.py --fusion EF --condition independent
python3 train_sigma_optuna.py --fusion CMA --condition independent
python3 train_sigma_optuna.py --fusion PoE --condition independent

# 2. Train Ablation Study Variants (w/o L_cross, w/o L_Cox, w/o Mask Input, Clinical-only, Genomic-only)
python3 train_sigma_ablation.py

# 3. Train Baseline Methods
python3 train_dl_baselines_optuna.py
python3 train_baselines_optuna.py
python3 train_gain_mida.py
```

### Step 4: Downstream Evaluation
```bash
# Evaluate Survival Prediction (RSF, XGBoost, DeepSurv, DeepHit)
python3 evaluate_survival.py --severity severe --condition independent

# Evaluate PAM50 Subtype Classification
python3 evaluate_pam50.py --condition independent

# Evaluate Ablation Variants (Survival C-index & PAM50 Accuracy)
python3 evaluate_ablation.py

# Evaluate Latent Integration (CCA & Silhouette) and Cross-Modal Translation (C2G / G2C)
python3 evaluate_integration.py
python3 evaluate_crossmodal.py
python3 evaluate_direct_survival.py
```

### Step 5: Figure Generation
```bash
python3 generate_figures.py
```

---

## 📜 License

This project is licensed under the MIT License.
