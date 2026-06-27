# Multi-Agent Cyber Threat Forecasting System

## Overview

This system implements a sophisticated multi-agent architecture for proactive cybersecurity threat forecasting, integrating dark web intelligence with explainable AI techniques. The platform employs seven specialised LLM-powered agents that collaborate to deliver end-to-end threat analysis, model selection, forecasting, and mitigation recommendations.

### Key Features

- **Multiple Forecasting Models**: From classical statistical methods (ARIMA, SARIMA) to cutting-edge deep learning (Graph Transformers, Vision Transformers, BMTGNN)
- **Multi-Agent Architecture**: Seven autonomous agents handling preprocessing, model selection, analysis, XAI, PAT recommendations, visualisation, and reporting
- **Dark Web Intelligence Integration**: Extends forecast lead-time through early signal detection
- **Explainable AI (XAI)**: multiple interpretability methods including SHAP, LIME, attention analysis, and counterfactual explanations
- **Pertinent Alleviation Technologies (PAT)**: Automated security control recommendations based on forecasted threats
- **Retrieval-Augmented Generation (RAG)**: Optional threat intelligence augmentation for enhanced context

## Prerequisites

### Required Dependencies

Ensure all dependency files are present in the same repository:

```
repository/
├── mainscriptv4_w_ALL_models_working_up_to_report_gen.py
├── ModelSelectionAgentII.py
├── hyperoptim_transfer_learning_rework.py
├── hyperparameter_optim_bmtgnn_transfer_learning.py
├── hyperparameter_optimization_ensemble_pretraining.py
├── hyperparams_optim_vanilla_graph.py
├── hyperparams_optim_vanilla_vit_transfer_learning.py
├── transfer_learning_hyperparams_optim.py
├── AnalysisAgent.py
├── XAIAnalyzer.py
├── ReportAgent.py
├── VisualizationAgent.py
├── PATRecommendationAgent.py
├── preprocess_agent.py
├── LSTMForecaster.py
├── RAGSystem.py
└── data/
    ├── sm_data_g.csv
    └── graph.csv
```

### Python Environment

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Core Libraries

```bash
pip install pandas numpy matplotlib seaborn
pip install torch scikit-learn statsmodels prophet
pip install langchain langchain-community
pip install shap lime dice-ml networkx pgmpy
pip install openai anthropic  # Optional: for cloud LLM providers
```

### Optional Dependencies

For advanced model architectures:

```bash
pip install torch-geometric  # Graph Neural Networks
pip install transformers     # Vision Transformers
```

---

## Installation

1. **Clone the repository** (or ensure all files are in the same directory)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up local LLM** (Ollama example):
   ```bash
   # Install Ollama: https://ollama.ai
   ollama pull deepseek-r1:8b
   ```

4. **Prepare data files**:
   - Place your cybersecurity time series data in `data/sm_data_g.csv`
   - Place graph topology metadata in `data/graph.csv`

---

## Configuration Guide

### Basic Configuration

The system is configured through the `ForecastConfig` dataclass in the main script. Modify lines 2700-2722:

```python
config = ForecastConfig(
    llm_provider = "ollama",              # LLM provider selection
    llm_model = "deepseek-r1:8b",         # Model name
    data_path = "data/sm_data_g.csv",     # Time series data
    graph_path = "data/graph.csv",        # Graph topology
    forecast_months = 36,                 # Forecast horizon
    output_dir = "forecast_results",      # Output directory
    enable_xai = True,                    # Enable XAI analysis
    enable_rag = False,                   # Enable RAG system
    enable_preprocessing = True,          # Enable advanced preprocessing
)
```

---

## Changing the LLM Provider

### Option 1: Local Models with Ollama (Default)

**Configuration**:
```python
config = ForecastConfig(
    llm_provider = "ollama",
    llm_model = "deepseek-r1:8b",         # Or: "llama2", "mistral", "codellama"
    base_url = "http://localhost:11434",  # Ollama server address
)
```

**Available Ollama Models**:
```bash
ollama pull deepseek-r1:8b      # Recommended: balanced performance
ollama pull llama2:13b          # Alternative: larger model
ollama pull mistral:7b          # Alternative: efficient model
```

### Option 2: OpenAI API

1. **Uncomment OpenAI support** in `LLMManager` class (lines 213-220)

2. **Configure**:
```python
config = ForecastConfig(
    llm_provider = "openai",
    llm_model = "gpt-4",                  # Or: "gpt-3.5-turbo"
    api_key = "your-openai-api-key",      # Or set OPENAI_API_KEY env var
)
```

### Option 3: Anthropic Claude API

1. **Uncomment Anthropic support** in `LLMManager` class (lines 221-226)

2. **Configure**:
```python
config = ForecastConfig(
    llm_provider = "anthropic",
    llm_model = "claude-3-opus-20240229",  # Or: "claude-3-sonnet-20240229"
    api_key = "your-anthropic-api-key",    # Or set ANTHROPIC_API_KEY env var
)
```

---

## Enabling/Disabling System Features

### Explainable AI (XAI) Analysis

**Enable XAI** (default: enabled):
```python
config = ForecastConfig(
    enable_xai = True,
    xai_methods = [
        "shap",              # SHAP value analysis
        "lime",              # Local interpretable explanations
        "attention",         # Attention weight visualisation
        "permutation",       # Permutation feature importance
        "counterfactual",    # Counterfactual explanations
        "causal",            # Causal network analysis
        "faithfulness",      # Explanation faithfulness metrics
        "dynamic_weights",   # Dynamic weight interpretation
        "dice",              # DiCE counterfactuals
        "consensus",         # Multi-model consensus
        "anchors"            # Rule-based explanations
    ]
)
```

**Disable XAI** (for faster execution):
```python
config = ForecastConfig(
    enable_xai = False,
)
```

**Selective XAI Methods** (choose specific techniques):
```python
config = ForecastConfig(
    enable_xai = True,
    xai_methods = ["shap", "lime", "attention"],  # Only these three methods
)
```

### Retrieval-Augmented Generation (RAG)

**Enable RAG** for threat intelligence augmentation:
```python
config = ForecastConfig(
    enable_rag = True,
    rag_db_path = "cyber_threat_db",  # Path to vector database
)
```

**Requirements for RAG**:
1. Ensure `RAGSystem.py` is in the repository
2. Prepare threat intelligence documents for vector database
3. The system will automatically initialise the RAG database on first run

**Disable RAG** (default):
```python
config = ForecastConfig(
    enable_rag = False,
)
```


### Advanced Preprocessing

**Enable LLM-guided preprocessing** (default: enabled):
```python
config = ForecastConfig(
    enable_preprocessing = True,
    preprocess_config = {
        'preprocess': {
            'outlier_threshold': 1.5,           # IQR multiplier for outlier detection
            'enable_llm_preprocessing': True    # Use LLM for strategy recommendation
        }
    }
)
```

**Disable advanced preprocessing** (use basic strategies):
```python
config = ForecastConfig(
    enable_preprocessing = False,
)
```

---

## Running the System

### Basic Execution

```bash
python mainscriptv4_w_ALL_models_working_up_to_report_gen.py
```

### Execution Flow

1. **Data Loading**: Loads time series and graph topology
2. **Preprocessing**: Cleans data, handles outliers, normalises features
3. **Time Series Analysis**: Analyses trends, seasonality, stationarity
4. **Model Selection**: Selects optimal models via comprehensive validation
5. **Model Training**: Trains selected models on historical data
6. **Forecasting**: Generates predictions for specified horizon
7. **XAI Analysis**: Produces explanations for model predictions (if enabled)
8. **PAT Recommendations**: Suggests mitigation strategies based on forecasts
9. **Visualisation**: Creates plots and dashboards
10. **Report Generation**: Compiles comprehensive intelligence report

### Expected Runtime

- **Basic run** (XAI disabled, 3 models): ~5-15 minutes
- **Full run** (XAI enabled, all models): ~30-90 minutes
- **With RAG**: Add 5-10 minutes for initial database creation


## Model Selection

The system supports several forecasting architectures. Available models are automatically detected based on installed dependencies:

### Always Available
- ARIMA, SARIMA
- Exponential Smoothing
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)
- Gradient Boosting
- LSTM Networks
- Prophet

### Optional (require additional dependencies)
- **BMTGNN** (Bayesian Multivariate Temporal Graph Neural Network)
  - Requires: `hyperparameter_optim_bmtgnn_transfer_learning.py`
- **Vision Transformers** (SimpleVisionTransformer, VisionTransformer)
  - Requires: `hyperparameter_optimization_ensemble_pretraining.py`
- **Graph Transformers** (SimpleGraphTransformer, GraphTransformer)
  - Requires: `pretrain_script.py`, `transfer_learning_hyperparams_optim.py`
- **Spatio-Temporal Ensemble**
  - Requires: `hyperparameter_optimization_ensemble_pretraining.py`

The `ModelSelectionAgent` automatically selects the best models based on:
- Data characteristics (trend, seasonality, stationarity)
- Available computational resources
- Historical performance on validation set

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure all dependency files are in the same directory as the main script.

```bash
# Check all required files exist
ls -la *.py
```

### Issue: "BMTGNN not available" warning

**Solution**: This is expected if optional model files are missing. The system continues with available models. To enable:

```bash
# Ensure these files exist:
# - hyperparameter_optim_bmtgnn_transfer_learning.py
```

### Issue: "LLM connection failed"

**Solution**: For Ollama, ensure the server is running:

```bash
# Check Ollama status
ollama list

# Start Ollama server
ollama serve
```

For cloud APIs, verify API keys:

```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Issue: "Out of memory" errors

**Solution**: Reduce model complexity or forecast horizon:

```python
config = ForecastConfig(
    forecast_months = 12,              # Reduce from 36
    enable_xai = False,                # Disable XAI temporarily
    xai_methods = ["shap", "lime"],    # Use fewer XAI methods
)
```

### Issue: JSON parsing errors from LLM

**Solution**: The system includes fallback strategies. However, if persistent:

1. Try a different LLM model (larger models are more reliable)
2. Reduce temperature for more deterministic outputs (edit `LLMManager`, line 207)

---

## Performance Optimization

### For Faster Execution

```python
config = ForecastConfig(
    enable_xai = False,                    # Skip XAI analysis
    enable_rag = False,                    # Disable RAG
    enable_preprocessing = False,          # Use basic preprocessing
    forecast_months = 12,                  # Shorter horizon
    validation_config = {
        'n_candidates': 2,                 # Fewer model candidates
        'cv_folds': 3                      # Fewer cross-validation folds
    }
)
```

### For Maximum Accuracy

```python
config = ForecastConfig(
    enable_xai = True,
    enable_rag = True,
    enable_preprocessing = True,
    forecast_months = 36,
    validation_config = {
        'n_candidates': 5,
        'k_models': 5,
        'cv_folds': 10,
        'optimization_method': 'grid_search'
    }
)
```

---

## Research Context

This system was developed to address critical gaps in cyber threat forecasting:

1. **Extended Lead-Time**: Integration of dark web intelligence to detect threats earlier
2. **Model Transparency**: Comprehensive XAI frameworks for security decision-making
3. **Architecture Comparison**: Systematic evaluation of deep learning approaches for cyber time-series
4. **Regulatory Compliance**: XAI implementations tailored for DORA and similar frameworks
5. **Actionable Intelligence**: PAT recommendations bridge prediction to mitigation

### Key Metrics

- **Forecast Lead-Time**: Extended through dark web signal integration
- **Prediction Accuracy**: Optimised via multi-model ensemble and architecture comparison
- **Interpretability**: 11 XAI methods for comprehensive explanation coverage
- **Operational Readiness**: PAT mapping provides actionable security controls


## Acknowledgements

- Builds upon Almahmoud (2023) work on Bayesian MTGNN for long-term forecasting
- Integrates threat intelligence concepts from MITRE ATT&CK framework
- XAI methodologies adapted from interpretable machine learning research


