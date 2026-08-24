# SIGMA: Survival Integration via Generative Multi-omics Alignment

Official implementation of **SIGMA**, a multimodal variational autoencoder designed to address **Missing Not At Random (MNAR)** mechanisms in heterogeneous clinical and genomic datasets within a survival-aware generative framework.

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

## 📂 Repository Structure

```text
├── Raw data/
│   └── scripts/
│       ├── sigma_model.py              # Core SIGMA model architecture, encoders, decoders, and loss functions
│       ├── preprocess_sigma.py         # METABRIC and TCGA clinical & genomic preprocessing
│       ├── calibrate_mnar.py           # Threshold-interaction MNAR simulation parameter grid search
│       ├── create_folds.py             # 5-fold cross-validation split generation
│       ├── train_sigma_optuna.py       # SIGMA training with Optuna hyperparameter optimization
│       ├── train_dl_baselines_optuna.py# Deep learning baseline runners (StandardVAE, MNARSVAE, MyeVAE)
│       ├── train_baselines_optuna.py   # Classical baseline runners (MICE, missForest, GAIN, MIDA)
│       ├── train_sigma_ablation.py     # 5 ablation variants under severe independent MNAR
│       ├── evaluate_survival.py        # Downstream survival models (RSF, XGBoost, DeepSurv, DeepHit)
│       ├── evaluate_pam50.py           # PAM50 molecular subtype classification
│       ├── evaluate_ablation.py        # Downstream evaluation for ablation variants
│       ├── evaluate_integration.py     # CCA and Silhouette score computation
│       ├── evaluate_crossmodal.py      # Cross-modal (C2G and G2C) reconstruction metrics
│       └── generate_figures.py         # Figure generation for benchmark results
├── elsarticle/                         # LaTeX manuscript source and TikZ architecture diagrams
├── requirements.txt                    # Dependencies
├── .gitignore
└── README.md
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

Place downloaded files in `Raw data/original_datasets/brca_metabric` and `Raw data/original_datasets/brca_tcga_pan_can_atlas_2018`.

---

## 🧪 Running the Pipeline

### Step 1: Preprocessing & Folds
```bash
cd "Raw data/scripts"
python3 preprocess_sigma.py
python3 create_folds.py
```

### Step 2: MNAR Calibration
```bash
python3 calibrate_mnar.py
```

### Step 3: Training & Hyperparameter Tuning (Optuna)
```bash
# Train SIGMA (EF, CMA, PoE)
python3 train_sigma_optuna.py --fusion EF --condition independent
python3 train_sigma_optuna.py --fusion CMA --condition independent
python3 train_sigma_optuna.py --fusion PoE --condition independent

# Train Baselines
python3 train_dl_baselines_optuna.py
python3 train_baselines_optuna.py

# Train Ablation Study Variants
python3 train_sigma_ablation.py
```

### Step 4: Downstream Evaluation
```bash
python3 evaluate_survival.py --severity severe --condition independent
python3 evaluate_pam50.py --condition independent
python3 evaluate_ablation.py
python3 evaluate_integration.py
python3 evaluate_crossmodal.py
```

---

## 📜 License

This project is licensed under the MIT License.
