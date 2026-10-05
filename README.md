# Enhancing cyber threat forecasting with dark web signals, transfer learning, an agentic LLM system and explainable AI

MSc Data Science dissertation, Birkbeck, University of London (2025). Author: Isobel (Bella) Smith.

This repository holds the code, processed data, trained model checkpoints and outputs for a project on proactive cyber threat forecasting. The project builds a monthly multivariate time series of cyber attacks and Pertinent Alleviation Technologies (PATs) for July 2011 to December 2024. It adds dark web and other external signals to that series and pre-trains forecasting models on unrelated time series (transfer learning). It then compares seven forecasting models, covering graph neural network, graph transformer, vision transformer and ensemble designs, and explains their forecasts with several XAI methods. An agentic LLM system chooses between models and writes a threat report.

The full write-up is in `MSc_Final_Project_Enhancing_Cyber_Threat_Forecasting_with_Dark_Web_Signals_and_Explainable_AI.pdf`. The slides are in `Presentation-of-Research-Proactive-Cyber-Threat-Forecasting.pdf`.

## Contents

1. [Overview and Key Contributions](#overview-and-key-contributions)
2. [Empirical Results and Analysis](#empirical-results-and-analysis)
3. [Explainable AI (XAI) Suite](#explainable-ai-xai-suite)
4. [Repository layout](#repository-layout)
5. [Choose how far back to start](#choose-how-far-back-to-start)
6. [Set up the environment](#set-up-the-environment)
7. [One-off step: path compatibility shim](#one-off-step-path-compatibility-shim)
8. [Stage 1: data collection and preparation](#stage-1-data-collection-and-preparation)
9. [Stage 2: transfer learning data](#stage-2-transfer-learning-data)
10. [Stage 3: models and XAI](#stage-3-models-and-xai)
11. [Stage 4: agentic LLM system](#stage-4-agentic-llm-system)
12. [Where the outputs go](#where-the-outputs-go)
13. [Known issues](#known-issues)
14. [Data sources, terms and ethics](#data-sources-terms-and-ethics)
15. [Citation and acknowledgements](#citation-and-acknowledgements)

## Overview and Key Contributions

1. **Expanded Intelligence Dataset**
   * Dataset expanded from 5 sources and 143 features to **11 sources and 2,126 features**, covering 13.5 years of time-series data through December 2024.
   * Six additional threat perspectives integrated:
     * Incident tracking (Hackmageddon, University of Maryland)
     * Dark web marketplace telemetry (pricing and volume for 67 attack categories across 14 Tor platforms)
     * Social and news discourse (RSS feeds, Bluesky)
     * Public search awareness (Google Trends across 86 security topics)
     * Vulnerability databases (NVD / CVE disclosures)
     * Geopolitical conflict metrics (ACLED data)

2. **Architectural Benchmarking**
   * **Bayesian MTGNN:** Relational graph modelling with probabilistic uncertainty (baseline model).
   * **Graph Transformer (GPS-enabled):** Combines local message passing with global attention mechanisms.
   * **Vision Transformer (ViT):** Formulates temporal time-series as image patches to capture long-range temporal dependencies.
   * **Spatiotemporal Ensemble:** Custom hybrid architecture fusing ViT and Graph Transformer branches via cross-attention and dynamic weighting.
   * **Agentic LLM:** Seven-agent pipeline adapted from TimeSeriesScientist, executed locally via Ollama and DeepSeek-R1 8B.

3. **Transfer Learning and Uncertainty Quantification**
   * Non-baseline models pre-trained across financial, epidemiological, and terrorism domains prior to cyber fine-tuning.
   * Integration of Monte Carlo (MC) dropout uncertainty for Graph and Vision Transformers.
   * Explainable AI layer incorporating SHAP, LIME, DiCE counterfactuals, and causal analysis.

4. **Novelty Statements**
   * First systematic integration of dark web marketplace signals into long-range cyber threat forecasting.
   * First application of Monte Carlo dropout uncertainty within Graph and Vision Transformers for this domain.
   * First specialized agentic LLM architecture closing the pipeline between forecast, explanation, and mitigation recommendation.

## Empirical Results and Analysis

### 1. Error Analysis

Evaluation across Relative Absolute Error (RAE) and Root Relative Squared Error (RRSE). Lower values represent higher predictive accuracy.

| Model Architecture | Average RAE | Average RRSE | Notes |
| :--- | :--- | :--- | :--- |
| **Original Paper (Almahmoud, 2025)** | 0.7700 | 0.8300 | Baseline reported in literature |
| **Baseline BMTGNN (Extended Data)** | 0.6370 | 0.8190 | Extended time horizon without dark web signals |
| **BMTGNN (+ Dark Web Data)** | 0.9944 | 0.9991 | Performance degraded under high feature dimensionality |
| **Graph Transformer** | 0.5040 | 0.5220 | GPS-enabled graph attention |
| **Vanilla Graph Transformer** | 0.4980 | 0.5560 | Standard graph transformer setup |
| **Vision Transformer** | 0.3887 | 0.7771 | Temporal patch-based modeling |
| **Vanilla Vision Transformer** | 0.3831 | 0.7659 | Simplified vision transformer setup |
| **Agentic LLM (DeepSeek-R1 8B)** | 0.2986 | 0.3467 | Seven-agent autonomous forecasting pipeline |
| **Spatiotemporal Ensemble** | **0.0258** | **0.0580** | **Lowest overall error rate across all models** |

#### Key Analytical Insights
* **Dataset Expansion Effect:** Updating the baseline BMTGNN model with extended time-series data through 2024 reduced RAE from 0.7700 to 0.6370, confirming the benefit of expanded temporal coverage.
* **BMTGNN Anomaly:** Introducing high-dimensional dark web data caused BMTGNN error rates to rise to 0.9944 (RAE) and 0.9991 (RRSE). This degradation suggests potential capacity or structural limits within adaptive graph construction when handling high-dimensional feature spaces.
* **Transformer Performance:** Transformer architectures showed enhanced capability in processing complex multi-source signals. The Vanilla Vision Transformer achieved an RAE of 0.3831, while the Agentic LLM reached 0.2986.
* **Ensemble Superiority:** The custom Spatiotemporal Ensemble delivered the best predictive accuracy overall, achieving an RAE of 0.0258 and an RRSE of 0.0580. This represents approximately a 30-fold error reduction compared to the original study.


### 2. Operational Readiness and Adversarial Robustness

Models were evaluated on operational suitability and resilience against malicious input manipulation:
* **Operational Readiness Index (ORI):** Composite metric balancing detection quality (precision, recall), operational burden (false positive rate), and execution constraints (resource efficiency).
* **Adversarial Robustness:** Measures resistance to adversarial perturbations designed to bypass or distort model predictions.

| Model Architecture | Operational Readiness Index (ORI) | Adversarial Robustness |
| :--- | :--- | :--- |
| **Baseline BMTGNN** | 0.4860 | 0.1070 |
| **BMTGNN (+ Dark Web Data)** | 0.5903 | 0.9259 |
| **Graph Transformer** | 0.5220 | 0.9260 |
| **Vanilla Graph Transformer** | 0.5560 | 0.9260 |
| **Vision Transformer** | 0.5100 | 0.9260 |
| **Vanilla Vision Transformer** | 0.4080 | 0.9260 |
| **Spatiotemporal Ensemble** | 0.7453 | 0.9261 |
| **Agentic LLM** | **0.8679** | 0.8719 |

#### Deployment Considerations
* **Adversarial Resilience:** Exposure to dark web data elevated adversarial robustness across all architectures from 0.1070 (Baseline) to approximately 0.9260. Training on naturally noisy, unstructured underground data appears to act as implicit adversarial hardening.
* **Operational Trade-Offs:** Although the Spatiotemporal Ensemble achieved the lowest prediction error (RAE 0.0258), its ORI score was 0.7453 due to higher computational overhead.
* **Optimal Deployment Balance:** The Agentic LLM achieved the highest ORI score (0.8679) alongside strong adversarial robustness (0.8719), making it particularly suited to environments where operational efficiency, low latency, and resource constraints are primary considerations.

## Explainable AI (XAI) Suite

To ensure predictions are interpretable and compliant with regulatory standards (such as DORA and SEC disclosure rules), the system integrates four XAI methodologies:
1. **SHAP (Global Importance):** Identifies primary driving signals across the feature space, highlighting Google Trends, RSS feeds, and honeypot data as key indicators.
2. **LIME (Local Explanations):** Explains individual point predictions, showing reliance on mid-window temporal context to capture threat build-up patterns.
3. **Monte Carlo Dropout (Uncertainty Quantification):** Computes confidence intervals to differentiate between low-uncertainty automated alerts and high-uncertainty signals requiring human intervention.
4. **DiCE & Causal Analysis:** Generates sparse counterfactuals to identify minimal required feature changes for alternative outcomes, separating genuine causal precursors from coincidental correlations.

## Repository layout

```
.
├── 1. Data Collection and Preparation/   # Stage 1 notebooks and the processed cyber dataset (Data/)
├── 2. Transfer Learning/                 # Stage 2 notebook and the processed pre-training datasets
├── 3. Models + XAI/
│   ├── Baseline BMTGNN/                  # Baseline Bayesian MTGNN (after Almahmoud, 2023)
│   ├── BMTGNN/                           # B-MTGNN with transfer learning and XAI
│   ├── Ensemble/                         # Spatio-temporal ensemble
│   ├── Graph Transformer/                # Graph Transformer with transfer learning
│   ├── Vanilla Graph Transformer/        # Simple Graph Transformer
│   ├── Vision Transformer/               # Vision Transformer with transfer learning
│   ├── Vanilla ViT/                      # Simple Vision Transformer
│   └── Agentic LLM/                      # Multi-agent forecasting and reporting system
├── Appendix/                             # requirements.txt, environment.yml, data dictionary,
│                                         # architecture diagrams, Docker files, prompt list
├── MSc_Final_Project_...pdf              # Dissertation
└── Presentation-of-Research-...pdf       # Slides
```

Each model folder is self-contained. It has its own `data/` and `pretrain_data/` folders, its own scripts, and a `model/` folder that the scripts write into. Most folders already contain the trained checkpoints and outputs used in the dissertation.

The repository has roughly 8,500 files, most of them plots. A shallow clone saves time:

```bash
git clone --depth 1 https://github.com/bgsbgsbgs42/MSc-Dissertation-Project.git
```

## Choose how far back to start

You do not have to rerun everything. The processed data and most trained models are committed, so pick the route that matches what you want to check.

| Route | What you do | What you need |
|---|---|---|
| A. Read the results | Browse the plots and reports in each model folder (see [Where the outputs go](#where-the-outputs-go)) | Nothing beyond a browser |
| B. Re-run the models | Start at [Stage 3](#stage-3-models-and-xai) using the committed `data/` and `pretrain_data/` files | Python environment, ideally a CUDA GPU |
| C. Rebuild the data | Start at [Stage 1](#stage-1-data-collection-and-preparation) | API keys, third-party downloads, and for the dark web step a hardened virtual machine |

Route C cannot reproduce the dataset exactly, because several sources change over time and the raw dark web data is not published.

## Set up the environment

The pinned dependencies are in `Appendix/requirements.txt`. `Appendix/environment.yml` records Python 3.11.13, and the pinned NumPy version (2.3.5) needs Python 3.11 or later.

### Recommended: pip in a fresh virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r Appendix/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

The extra index is needed because `requirements.txt` pins a CUDA 12.6 build of `torchvision` (`torchvision==0.24.1+cu126`). On a machine without an NVIDIA GPU, change that line to `torchvision==0.24.1` and drop the extra index.

### Alternative: conda

```bash
conda env create -f Appendix/environment.yml
conda activate combined-environment
```

`environment.yml` was exported from a working machine. It mixes Python 3.11 and 3.12 package builds, so conda may fail to solve it. If it does, use the pip route.

### Extra tools for specific stages

| Stage | Extra requirement | Notes |
|---|---|---|
| Stage 1, Elsevier cells | Firefox and geckodriver | Used through Selenium |
| Stage 1, dark web notebook | Tor and the `stem` package | `stem` is not in `requirements.txt`: `pip install stem` |
| Stage 4, agentic system | [Ollama](https://ollama.com) with `deepseek-r1:8b` | `ollama pull deepseek-r1:8b`, then `ollama serve` |

### GPU and runtime

Every training script picks `cuda` when it is available and falls back to the CPU otherwise. The two baseline scripts are the exception.


### Docker

`Appendix/Docker Files/` holds a Dockerfile and compose files. They do not build as committed, because the Dockerfile copies `requirements.txt` from its own folder and its `CMD` points to a script path that is not in this repository. Treat them as a record of the original set-up.

## One-off step: path compatibility shim

The scripts were written in a local folder called `Dissertation/`, where several model folders had different names. Many scripts still use those old paths, for example `Dissertation/Bayesian MTGNN/data/sm_data_g.csv`. Other scripts in the same folder use relative paths such as `./data/sm_data_g.csv`.

Both kinds of path work if you run each script from inside its model folder and add a symbolic link that maps the old name back onto that folder. Do this once per folder you plan to use.

| Folder in this repository | Old name used in the code |
|---|---|
| `3. Models + XAI/BMTGNN` | `Bayesian MTGNN` |
| `3. Models + XAI/Baseline BMTGNN` | `BMTGNN Baseline` |
| `3. Models + XAI/Ensemble` | `Ensemble Variant` |
| `3. Models + XAI/Graph Transformer` | `Graph Transformer` |
| `3. Models + XAI/Vanilla Graph Transformer` | `Vanilla Graph Transformer` |
| `3. Models + XAI/Vision Transformer` | `Vision Transformer` |
| `3. Models + XAI/Agentic LLM` | `Agentic LLM` |

macOS or Linux, run from the repository root:

```bash
make_shim () {   # usage: make_shim "<folder>" "<old name>"
  ( cd "3. Models + XAI/$1" && mkdir -p Dissertation && ln -sfn .. "Dissertation/$2" )
}
make_shim "BMTGNN"                    "Bayesian MTGNN"
make_shim "Baseline BMTGNN"           "BMTGNN Baseline"
make_shim "Ensemble"                  "Ensemble Variant"
make_shim "Graph Transformer"         "Graph Transformer"
make_shim "Vanilla Graph Transformer" "Vanilla Graph Transformer"
make_shim "Vision Transformer"        "Vision Transformer"
make_shim "Agentic LLM"               "Agentic LLM"
```

Windows (Command Prompt with Developer Mode on, or as administrator), one folder at a time:

```bat
cd "3. Models + XAI\BMTGNN"
mkdir Dissertation
mklink /D "Dissertation\Bayesian MTGNN" ..
```

After this, `Dissertation/Bayesian MTGNN/data/sm_data_g.csv` and `./data/sm_data_g.csv` point to the same file. The shim is a workaround. Replacing the hard-coded paths in the code would remove the need for it.

## Stage 1: data collection and preparation

Folder: `1. Data Collection and Preparation/`

Skip this stage unless you want to rebuild the cyber dataset. Its outputs are already in `Data/`.

The notebooks write to the current working directory, so start Jupyter from inside this folder. Several cells contain hard-coded input paths that you will need to edit.

### 1a. `Data Collection.ipynb`

Run the cells in this order. Each cell is independent, so you can skip a source you do not have access to.

| Cell | Source | Before running | Output |
|---|---|---|---|
| 0 | Public holidays (`holidays` package) | Nothing | `PH_2011_2024.csv` |
| 1 | Elsevier, attack mentions | Put your Elsevier API key in `config.json` | `Attacks_NoM.txt` |
| 3 | Elsevier, PAT mentions | Same key | `PTs_NoM.txt` |
| 5 | Security RSS feeds | Nothing | `security_RSS_vulnerabilities_monthly.csv`, `security_RSS_solutions_monthly.csv` |
| 6 | CISSM Cyber Events Database | Download the database from https://cissm.umd.edu/research-impact/publications/cyber-events-database-home and place it in this folder | `cyber_attacks_summary_by_type.csv` |
| 8 | ACLED political violence counts | Download the ACLED country-month file and place it in this folder | `political_violence_by_country_2017_2024.csv` |
| 10 | Bluesky war and conflict posts | Enter a Bluesky username and password in the cell | `ACA_bluesky.csv` |
| 12 | CVE records | Clone https://github.com/CVEProject/cvelistV5 and set `data_directory` to its location | `monthly_counts.csv`, `unique_mentions.txt`, `summary_stats.txt` |

Do not commit `config.json` with a key in it, or the Bluesky cell with credentials in it.

### 1b. `Hackmaggedon formatting.ipynb`

Hackmageddon data needs permission from the site owner before download.

1. Cell 0 reads the downloaded file. Edit `input_path` to point to it. Output: `NoI_daily_2024.csv`.
2. Cell 2 converts daily counts to monthly. Output: `NoI_monthly_2024.csv`.
3. Cell 4 reorders the columns. Output: `hackmageddon_reordered_output.csv`.

### 1c. `System Hardening and Dark Web Data Collection.ipynb`

The notebook's own guidance applies. Run it only inside an isolated virtual machine such as Whonix or Tails, connect to a VPN before Tor, and avoid root except for the hardening cell. Replace the placeholder onion addresses with your approved research targets and adjust the page selectors to match them.

1. Cell 0 hardens the machine. It needs root.
2. Cell 2 collects forum and market pages over Tor and saves one JSON file per source.
3. Cell 3 extracts dates, site names and term counts from a folder of scraped CSV files. Output: `scraping_results.csv`.
4. Cell 5 cleans the 2020 to 2024 market data. Outputs: `monthly_term_counts.csv`, `monthly_average_prices.csv`.

The raw dark web data is not included in this repository.

### 1d. Merge the sources

The monthly outputs from 1a to 1c are combined into one table, `Data/CT-0711-1224.csv`. It has 162 months (July 2011 to December 2024) and about 2,100 series covering attack types by country, PAT mentions, conflict counts and holidays. `Appendix/Data Dictionary.md` describes every column.

### 1e. `Data Preparation.ipynb`

1. Cell 0 imputes missing values. Put `CT-0711-1224.csv` in the working directory and run it. It writes `imputed_data_2011-2024.csv`, but the cells that follow expect `imputed_cybersecurity_data.csv`.
2. Cell 2 is optional. It plots the imputation against the original data and saves `imputation_report.png`.
3. Cell 4 is optional. It applies SMOTE to `imputed_cybersecurity_data.csv` and saves `cyber_threats_smote_augmented.csv` with summary tables.
4. Cell 6 applies double exponential smoothing (alpha 0.1, beta 0.3) to `imputed_cybersecurity_data_for smoothing.txt` and writes `sm_data.csv`. The input is the imputed table saved as tab-separated values with no header or date column.

`sm_data_g.csv` is `sm_data.csv` with the column headers restored [check]. The models read `sm_data_g.csv` and `graph.csv`. `graph.csv` lists each attack series in its first column, followed by the PAT series linked to it.

### 1f. Optional analysis

`Data Analysis (Optional).ipynb` produces attack and solution correlations, feature importance rankings and a descriptive XAI analysis of the data. It reads from `./data/`, but the folder here is `Data/`. Rename the folder or edit the path on case-sensitive file systems such as Linux. `Data/datadirectanalyserxai.py` is a standalone version of the third cell.

## Stage 2: transfer learning data

Folder: `2. Transfer Learning/`

Skip this stage unless you want to rebuild the pre-training data. The processed files are in this folder, and copies are already in each model's `pretrain_data/` folder.

The models are pre-trained on six unrelated monthly time series before they are fine-tuned on the cyber data. Run the cells of `Transfer Learning Data Collection + Preparation.ipynb` in order. Most cells contain a hard-coded input path beginning `Dissertation/Transfer Learning Data/` that you will need to change, and cell 7 contains an absolute Windows path.

| Cell | Dataset | Input you supply | Output |
|---|---|---|---|
| 0 | Yahoo Finance stock prices | A ticker list, `stock_profile_data.csv` (not included; the markdown cell explains the fallback list) | `monthly_stock_prices_2011_2024.csv` |
| 2 | Clean the stock file | Output of cell 0 | `cleaned_yfinance_output.csv` |
| 4 | Impute stock prices | Output of cell 2 | `imputed_yfinance_data.csv` |
| 5 | ONS monthly GDP | `ONS Monthly gross domestic product.csv` | `imputed_ONS_data.csv` |
| 6 | FRED series | A folder of FRED CSV downloads | `financial_data_1975.csv`, `financial_data_2000.csv` |
| 7 | Hybrid imputation of the FRED files | Run once per output of cell 6 | `imputed_2000_FRED_data_hybrid.csv`, `imputed_1975_FRED_data_hybrid.csv` |
| 9 | FRED daily to monthly | `imputed_1975_FRED_data_hybrid.csv` | `monthly_averages_FRED_1975.csv` |
| 10 | Johns Hopkins COVID-19 deaths | `john_hopkins_time_series_covid19_deaths_global.csv` | `transposed_JH_covid_deaths_global.csv` |
| 11 | Global Terrorism Database | `globalterrorismdb_0522dist.csv` | `monthly_attack_summary_global_terrorism_database.csv` |
| 12 | Smooth every CSV in a folder | Set `input_folder` to the folder holding the files above | One `sm_g<original name>.csv` per file |

The `sm_g...` files are the ones the models use. Copy them into the `pretrain_data/` folder of each model you want to retrain.

## Stage 3: models and XAI

Folder: `3. Models + XAI/`

### Rules that apply to every model

1. Complete the [path compatibility shim](#one-off-step-path-compatibility-shim) for the folder.
2. `cd` into the model folder and run every script from there.
3. Run the scripts in the order given. Each script reads files the previous one wrote, mostly under `transfer_learning_pretrain/` and `model/`.
4. Quote file names that contain spaces, for example `python "xai integration.py"`.
5. Committed checkpoints are overwritten when you retrain. Copy the `model/` folder first if you want to keep the dissertation versions.

The models do not depend on each other, so you can run one without the others.

### Which data each model uses

Three versions of `sm_data_g.csv` are in the repository. Use the copy inside the model folder, not the Stage 1 file.

| Version | Shape (months × series) | Used by |
|---|---|---|
| Stage 1 output, `1. Data Collection and Preparation/Data/` | 162 × 2,125 (country-level) | Not read directly by any model |
| Main model data | 162 × 645 (global aggregates) | BMTGNN, Graph Transformer, Vision Transformer, Vanilla ViT |
| Baseline data | 138 × 123 | Baseline BMTGNN, Vanilla Graph Transformer |


### Baseline BMTGNN

Folder: `3. Models + XAI/Baseline BMTGNN/`. This is the Bayesian MTGNN baseline that the other models are compared against. `net.py`, `layer.py`, `trainer.py` and `util.py` are the model's building blocks and are imported by the scripts below rather than run.

The order below is inferred from the scripts' inputs and outputs [check]:

```bash
cd "3. Models + XAI/Baseline BMTGNN"
python smoothing.py                       # optional: data/data.txt -> data/sm_data.csv
python train_test.py --device cpu         # hyperparameter search -> model/Bayesian/hp.txt, model.pt
python train.py --device cpu              # final model -> model/Bayesian/o_model.pt
python evaluate.py
python forecast.py
python pt_plots.py
```

Both training scripts default to `--device cuda:1`, a second GPU. Pass `--device cpu` or `--device cuda:0`.

`Comprehensive Script Baseline.ipynb` covers the same four steps (search, final training, evaluation, forecast) in four cells.

### BMTGNN

Folder: `3. Models + XAI/BMTGNN/`

```bash
cd "3. Models + XAI/BMTGNN"
python transfer_learning_pretrain.py                  # pre-train on pretrain_data/, fine-tune on data/
python hyperparameter_optim_bmtgnn_transfer_learning.py
python model_training_bmtgnn_no_forecast.py           # final model -> model/o_model.pt
python forecast_w_pretrain.py
python evaluation.py
python "xai integration.py"
python pt_plotting.py
```

`train_o_model_w_o_model.py` is not part of this sequence. Before running `xai integration.py`, check its model path (see [Known issues](#known-issues)).

### Ensemble

Folder: `3. Models + XAI/Ensemble/`

This folder has no `data/` folder and no trained checkpoint, so set up the data and train from scratch.

```bash
cd "3. Models + XAI/Ensemble"
mkdir -p data
cp ../BMTGNN/data/sm_data_g.csv ../BMTGNN/data/graph.csv data/    

python pretrain_transfer_learning.py
python hyperparameter_optimization_ensemble_pretraining.py \
    --pretrained_model transfer_learning_pretrain/transferred_model_for_hp_search.pt
python train_final_ensemble.py        # -> model/Ensemble/best_hyperparameter_model.pt
python forecast.py
python evaluation.py
python "xai integration.py"
python pt_plot.py
```

The `--pretrained_model` flag is needed because the script's default path begins with `/`, which points to the root of the file system.

### Graph Transformer

Folder: `3. Models + XAI/Graph Transformer/`

```bash
cd "3. Models + XAI/Graph Transformer"
python transfer_learning_pretrain_graph.py
python transfer_learning_hyperparams_optim.py
python train_operational_model_graph_w_model_file.py   # -> model/GraphTransformer/o_model.pt
python forecast_w_pretrain.py
python evaluation.py
python "xai integration.py"                            # check the model path first (Known issues)
python "xai (extra script).py"                         # optional
python pt_plot.py
```

### Vanilla Graph Transformer

Folder: `3. Models + XAI/Vanilla Graph Transformer/`. The order below is inferred from the scripts' inputs and outputs [check]. There are two hyperparameter scripts, and both write `model/SimpleGraphTransformer/best_hyperparameters.json`, so run one of them.

```bash
cd "3. Models + XAI/Vanilla Graph Transformer"
python "pretrain script.py"
python pretrain_hyperparams_optim.py        # with pre-training
# or: python hyperparams_optim_vanilla_graph.py   (without pre-training)
python train.py                             # -> model/SimpleGraphTransformer/final_model.pt
python "evaluate with mae, mape, rrse, rmse and rae.py"
python forecast.py
```


### Vision Transformer

Folder: `3. Models + XAI/Vision Transformer/`. The folder's own `readme.md` gives this order, but three file names in it differ from the files. The names below match the repository.

```bash
cd "3. Models + XAI/Vision Transformer"
python transfer_learning_pretrain.py
python hyperoptim_transfer_learning_rework.py
python train_final_model_vit_w_open_pt_file.py         # -> model/ViT/o_model.pt
python forecast_with_pretraining.py
python "evaluate_w_mape_and_mae rmse, rae, rrse.py"
python "xai script 1.py"
python "xai script 2.py"
python pt_plot.py
```

The final training script continues training from `final_transfer_model/final_model.pt` and saves `model/ViT/o_model.pt`. The forecast, evaluation and XAI scripts load `final_transfer_model/final_model.pt`, not the new file.

### Vanilla ViT

Folder: `3. Models + XAI/Vanilla ViT/`. The order below is inferred from the scripts' inputs and outputs [check].

```bash
cd "3. Models + XAI/Vanilla ViT"
python transfer_learning_pretrain.py                   # -> transfer_learning_pretrain/transferred_model_finetuned.pt
python hyperparams_optim_vanilla_vit_transfer_learning.py   # -> model/ViT_Transfer_Optim/hp.txt
python train.py                                        # -> model/ViT_Final/o_model.pt
python evaluate_w_mape_and_mae.py
python forecast.py
```

`forecast.py` writes its outputs under `model/Bayesian/forecast/`, a folder name carried over from the baseline code.

### Re-running evaluation from the committed checkpoints

Most folders already hold the trained models, so you can skip the pre-training, search and final training steps and run only the forecast, evaluation and plotting scripts. The Ensemble folder is the exception, because its checkpoint is not committed.

## Stage 4: agentic LLM system

Folder: `3. Models + XAI/Agentic LLM/`

A set of LLM-driven agents runs the whole forecasting workflow: preprocessing, time series analysis, model selection, training, forecasting, XAI, PAT recommendations, visualisation and a written report. The folder's own `README.md` documents the configuration options in detail.

1. Start Ollama and pull the default model:

   ```bash
   ollama pull deepseek-r1:8b
   ollama serve
   ```

2. Create the data folder the script expects:

   ```bash
   cd "3. Models + XAI/Agentic LLM"
   mkdir -p data
   cp ../BMTGNN/data/sm_data_g.csv ../BMTGNN/data/graph.csv data/    
   ```

3. Optional, for retrieval-augmented generation:
   1. Put the raw threat intelligence files (MITRE ATT&CK tables and the `cvelistV5` JSON files) in `RAG data/`.
   2. Run `RAG preparation.ipynb`. Its second cell writes `data/unified_embeddings.csv`.
   3. Set `enable_rag=True` in the configuration.

4. Edit the configuration near the end of the main script (the `ForecastConfig(...)` call at about line 3095) to choose the LLM provider, model, forecast horizon and which XAI methods to run.

5. Run the main script:

   ```bash
   python mainscriptv4_w_ALL_models.py
   ```

The folder's `README.md` names the main script `mainscriptv4_w_ALL_models_working_up_to_report_gen.py` and places the configuration at lines 2700 to 2722. The file in this repository is `mainscriptv4_w_ALL_models.py`, and the configuration is at about line 3095.

The main script imports the model modules in the same folder (`hyperparameter_optim_bmtgnn_transfer_learning.py`, `pretrain_script.py` and others). If one is missing, that model is skipped with a warning. The folder's README reports a run time of about 5 to 15 minutes without XAI and 30 to 90 minutes with all models and XAI.

Results go to the folder set by `output_dir` in the configuration. The results from the dissertation run are committed in `enhanced_cybersecurity_forecast_results/` and `reports/`.

## Where the outputs go

Inside each model folder:

| Path | Contents |
|---|---|
| `transfer_learning_pretrain/` | Pre-trained weights per source dataset and the fine-tuned starting model |
| `model/<Name>/hp.txt`, `*.json` | Chosen hyperparameters and search logs |
| `model/<Name>/Validation/`, `Testing/` | Per-series predicted against actual plots (PNG and PDF) and metrics (TXT) |
| `model/<Name>/forecast/plots/` | 36-month forecasts for each attack type |
| `model/<Name>/forecast/pt_plots/` | Forecasts for the PATs linked to each attack |
| `model/<Name>/forecast/gap/` | Per-attack gap files |
| `model/<Name>/forecast/data/` | Forecast values as text files |
| `xai outputs/`, `xai_outputs/`, `xai_results/` | Feature importance, attention, causal network, LIME, DiCE counterfactual and other XAI outputs |
| `*_evaluation_report.txt` | Summary metrics for the model |

The architecture diagrams are in `Appendix/Architecture Diagrams/`. `Appendix/Evaluation Framework Explanation.pdf` and `Appendix/Dataset Augmentation Explanation.pdf` explain the evaluation design and the data augmentation.

## Known issues

These were found while writing this README. None of them stops the code from running once the workarounds above are applied.

1. Many scripts use hard-coded `Dissertation/<old folder name>/` paths. The [shim](#one-off-step-path-compatibility-shim) handles them.
2. `BMTGNN/xai integration.py` and `Graph Transformer/xai integration.py` load `Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt`, the ensemble checkpoint. 
3. `Ensemble/hyperparameter_optimization_ensemble_pretraining.py` has a default `--pretrained_model` path beginning with `/`. Pass the path explicitly, as shown above.
4. `Baseline BMTGNN/train.py` and `train_test.py` default to `--device cuda:1`.
5. The `Ensemble` and `Agentic LLM` folders have no `data/` folder.
6. `Vision Transformer/readme.md` lists three file names that do not match the files.
7. `Agentic LLM/README.md` names a main script that does not exist under that name and gives out-of-date line numbers.
8. In `Data Preparation.ipynb`, the imputation cell's output name does not match the next cell's input name.
9. `Data Analysis (Optional).ipynb` reads `./data/` but the folder is `Data/`.
10. `stem`, needed for the dark web notebook, is missing from `requirements.txt`.
11. The Docker files do not build as committed.

## Data sources, terms and ethics

`Appendix/Data Dictionary.md` lists the sources: Hackmageddon, Elsevier, the University of Maryland CISSM Cyber Events Database, the National Vulnerability Database and CVE records, dark web markets and forums, 138 security RSS feeds, Google Trends, Twitter, Bluesky, ACLED and the Python `holidays` package. The transfer learning data comes from Yahoo Finance, the ONS, FRED, Johns Hopkins University's COVID-19 series and the Global Terrorism Database.

The Google Trends and Twitter series have no collection script in this repository.

Each third-party source has its own terms of use. This repository includes processed, aggregated monthly counts and does not include raw records. Anyone rebuilding the data must get their own access, and Hackmageddon requires the site owner's permission.

The dark web collection ran in an isolated, hardened virtual machine. Only aggregated term counts and average prices are published. No raw posts, usernames or other personal data are included.

## Citation and acknowledgements

### Citation

```bibtex
@mastersthesis{smith2025cyberforecasting,
  author = {Smith, Isobel},
  title  = {Enhancing Cyber Threat Forecasting with Dark Web Signals, Transfer Learning,
            an Agentic LLM System and Explainable AI},
  school = {Birkbeck, University of London},
  year   = {2025},
  type   = {MSc dissertation}
}
```

### Acknowledgements

- The baseline model and its forecasting approach build on Almahmoud's (2025) Bayesian MTGNN for long-term cyber threat forecasting [Almahmoud, Zaid & Yoo, Paul D. & Damiani, Ernesto & Choo, Kim-Kwang Raymond & Yeun, Chan Yeob, 2025. "Forecasting Cyber Threats and Pertinent Mitigation Technologies," Technological Forecasting and Social Change, Elsevier, vol. 210(C)​](https://github.com/zaidalmahmoud/Cyber-trend-forecasting)
- Agent System Framework: Adapted from [TimeSeriesScientist](http://arxiv.org/abs/2510.01538)  
- The PAT recommendations draw on the MITRE ATT&CK framework.
- Baseline data comes from [Hackmageddon](https://www.hackmageddon.com/)
- I'd also like to thank my supervisor, Dr Paul Yoo. 

Contact: Isobel (Bella) Smith, iggyggsmith42@gmail.com
