import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import shap
import lime
import lime.lime_tabular
import dice_ml as dice
import networkx as nx
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import openai
from langchain_core.messages import HumanMessage, SystemMessage
#from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pgmpy.estimators import PC, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import scipy.stats as stats
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict
from typing_extensions import Annotated, TypedDict
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import pickle
import joblib
from itertools import product
import time

try:
    from hyperparameter_optimization_ensemble_pretraining import (
        SpatioTemporalEnsemble, VisionTransformerForTimeSeries, 
        CyberThreatDataset, set_random_seed
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("Ensemble models not available - ensure hyperparameter_optimization_ensemble_pretraining.py is accessible")

try:
    from hyperparameter_optim_bmtgnn_transfer_learning import (
        gtnet, DataLoaderS, Optim, set_random_seed
    )
    BMTGNN_AVAILABLE = True
except ImportError:
    BMTGNN_AVAILABLE = False
    print("B-MTGNN not available - ensure hyperparameter_optim_bmtgnn_transfer_learning.py is accessible")

try:
    from pretrain_script import SimpleGraphTransformer, SimpleDataLoader
    from transfer_learning_hyperparams_optim import GraphTransformer, CyberThreatDataLoader
    GRAPH_TRANSFORMERS_AVAILABLE = True
except ImportError:
    GRAPH_TRANSFORMERS_AVAILABLE = False
    print("Graph Transformer models not available - ensure pretrain_script.py and transfer_learning_hyperparams_optim.py are accessible")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TypedDict for model selection output
class SelectedModel(TypedDict):
    """A selected model with its hyperparameters and reasoning."""
    model: Annotated[str, ..., "Name of the selected model"]
    hyperparameters: Annotated[Dict[str, Any], ..., "Hyperparameters for the model"]
    reason: Annotated[str, ..., "Reason for selecting this model"]

class ModelSelectionOutput(TypedDict):
    """Output format for model selection."""
    selected_models: Annotated[List[SelectedModel], ..., "List of selected models with hyperparameters"]

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


@dataclass
class ForecastConfig:
    """Configuration for the forecasting system"""
    # LLM Configuration
    llm_provider="ollama",
    llm_model="deepseek-r1:8b",  # or "mistral", "llama2", "codellama", etc.
    base_url="http://localhost:11434" # Ollama's default URL
    api_key: Optional[str] = None
    
    # Data Configuration
    data_path: str = "sm_data_g.csv"
    graph_path: str = "graph.csv"
    forecast_months: int = 36
    target_columns: List[str] = None
    
    # Model Configuration
    validation_split: float = 0.8
    seasonal_period: int = 12
    
    # Output Configuration
    output_dir: str = "forecast_results"
    plot_style: str = "seaborn"
    
    # Agent Configuration
    enable_agents: bool = True
    max_retries: int = 3
    
    # XAI Configuration
    enable_xai: bool = True
    xai_methods: List[str] = None
    
    # RAG Configuration
    enable_rag: bool = True
    rag_db_path: str = "cyber_threat_db"
    
    # New Models
    enable_lstm: bool = True
    enable_prophet: bool = True
    enable_validation: bool = True  
    validation_config: Dict[str, Any] = { #change
            'n_candidates': 3,
            'k_models': 3, 
            'cv_folds': 5,
            'optimization_method': 'grid_search'
        },
    
    # Preprocessing Configuration
    enable_preprocessing: bool = True
    preprocess_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize target columns after object creation"""
        if self.target_columns is None:
            # This will be populated after loading graph.csv
            self.target_columns = []
            
        # Initialize preprocessing configuration
        if self.preprocess_config is None:
            self.preprocess_config = {
                'preprocess': {
                    'outlier_threshold': 1.5,
                    'enable_llm_preprocessing': True
                }
            }
            
        # Initialize XAI methods with all available methods
        if self.xai_methods is None:
            self.xai_methods = [
                "shap",           # SHAP analysis
                "lime",           # LIME analysis  
                "attention",      # Attention visualization
                "permutation",    # Permutation feature importance
                "counterfactual", # Counterfactual analysis
                "causal",         # Causal network analysis
                "faithfulness",   # Explanation faithfulness
                "dynamic_weights", # Dynamic weight interpretation
                "dice",           # DiCE counterfactuals
                "consensus",      # Multi-model consensus
                "anchors"         # Rule-based explanations
            ]

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super().default(obj)
        
class LLMManager:
    """LLM-agnostic manager for different model providers"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.config.llm_provider == "ollama":
            return Ollama(
                model=self.config.llm_model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=0.1
            )
            print(f"LLM initialized successfully with model: {self.config.llm_model}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
        """elif self.config.llm_provider == "openai":
            if self.config.api_key:
                openai.api_key = self.config.api_key
            return ChatOpenAI(
                model_name=self.config.llm_model,
                temperature=0.1,
                max_tokens=4000
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=0.1,
                max_tokens=4000
            )"""
        
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the configured LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

class DataAgent:
    """Agent for data preprocessing and analysis"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
    def load_and_validate_data(self, data_path: str, graph_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Load and validate cybersecurity data"""
        logger.info("Loading cybersecurity data...")
        
        # Load main data
        data = pd.read_csv(data_path)
        logger.info(f"Data shape: {data.shape}")
        
        # Load graph metadata for node names 
        graph_data = pd.read_csv(graph_path)
        target_columns = graph_data.columns.tolist()  # columns graph.csv are targets
        logger.info(f"Target columns: {target_columns}")
        
        # Validate data dimensions match target columns
        if data.shape[1] != len(target_columns):
            logger.warning(f"Data columns ({data.shape[1]}) don't match target columns ({len(target_columns)})")
        
        return data, target_columns
    
    def preprocess_data(self, data: pd.DataFrame, node_names: List[str]) -> Dict[str, Any]:
        """Preprocess data using LLM-guided strategies"""
        system_prompt = """You are a cybersecurity data preprocessing expert. Analyse the data characteristics and recommend preprocessing strategies."""
        
        user_prompt = f"""
        Given cybersecurity time series data with shape {data.shape} and node names: {node_names[:5]}... (showing first 5),
        recommend preprocessing strategies for:
        1. Missing value handling
        2. Outlier detection and treatment
        3. Data normalization
        4. Feature engineering for cybersecurity context
        
        Please give detailed and verbose reasoning for each choice.
        
        Return JSON format:
        {{
            "missing_value_strategy": "str",
            "outlier_strategy": "str", 
            "normalization_method": "str",
            "feature_engineering": ["str"],
            "reasoning": "str"
        }}
        """
        
        llm_response = self.llm_manager.generate_response(system_prompt, user_prompt)
        strategies = self._parse_json_response(llm_response)
        
        # Apply preprocessing
        processed_data = self._apply_preprocessing(data, strategies)
        
        return {
            "data": processed_data,
            "strategies": strategies,
            "node_names": node_names
        }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback strategies
        return {
            "missing_value_strategy": "forward_fill",
            "outlier_strategy": "iqr",
            "normalization_method": "standard",
            "feature_engineering": ["rolling_mean_3", "rolling_std_3"],
            "reasoning": "Fallback strategies for cybersecurity data"
        }
    
    def _apply_preprocessing(self, data: pd.DataFrame, strategies: Dict[str, Any]) -> pd.DataFrame:
        """Apply preprocessing strategies"""
        processed_data = data.copy()
        
        # Handle missing values
        if strategies["missing_value_strategy"] == "forward_fill":
            processed_data = processed_data.ffill().bfill()
        
        # Outlier detection
        if strategies["outlier_strategy"] == "iqr":
            for col in processed_data.columns:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data[col] = np.clip(processed_data[col], lower_bound, upper_bound)
        
        # Normalization
        if strategies["normalization_method"] == "standard":
            scaler = StandardScaler()
            processed_data.iloc[:, 1:] = scaler.fit_transform(processed_data.iloc[:, 1:])
        
        return processed_data


class ModelSelectionAgent:
    """Enhanced Agent for model selection, hyperparameter optimization, and validation"""
    
    def __init__(self, llm_manager, config: Union[ForecastConfig, Dict[str, Any], None] = None):
        self.llm_manager = llm_manager
        self.available_models = [
            "ARIMA", "ExponentialSmoothing", "RandomForest", "LinearRegression", 
            "Prophet", "LSTM", "SARIMA", "SVR", "GradientBoosting"  " BMTGNN",
            "VisionTransformer", "SimpleVisionTransformer","SpatioTemporalEnsemble", "VisionTrasformerforTimeSeries",
            "SimpleGraphTransformer", "GraphTransformer"  
        ]
        self.config = config or {}
        self.n_candidates = self.config.get('n_candidates', 3)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.k_models = self.config.get('k_models', 3)
        self.optimization_method = self.config.get('optimization_method', 'grid_search')
        self.validation_split: float = 0.8
        if config is None:
            self.config: ForecastConfig = ForecastConfig()
        elif isinstance(config, ForecastConfig):
            self.config = config
        else:
             # Assume dict-like; unpack into ForecastConfig
             self.config = ForecastConfig(**config)
             
        self.n_candidates = getattr(self.config, "n_candidates", 3)
        self.cv_folds = getattr(self.config, "cv_folds", 5)
        self.k_models = getattr(self.config, "k_models", 3)
        self.optimization_method = getattr(self.config, "optimization_method", 'grid_search')
        self.validation_split: float = self.config.validation_split

        # Model storage
        self.trained_models = {}
        self.scalers = {}
        self.validation_results = {}

    def _ensure_dataframe(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for key in ("data", "processed_data", "df"):
                candidate = data.get(key)
                if isinstance(candidate, pd.DataFrame):
                    return candidate
        raise TypeError("Expected pd.DataFrame or dict containing a DataFrame")

    def run_validation(self, analysis_result: Dict[str, Any], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run complete validation pipeline - main entry point matching ValidationAgent"""
        logger.info("Starting comprehensive model validation process...")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        try:
            data_df = self._ensure_dataframe(data)
            # Get model selection from LLM
            model_selection = self._select_models_with_llm(analysis_result, data_df)
            
            if not model_selection or not model_selection.get('selected_models'):
                logger.warning("No models selected by LLM, using fallback selection")
                model_selection = self._get_fallback_models()
                
            selected_models = model_selection['selected_models']
            logger.info(f"LLM selected {len(selected_models)} models")
            
            # Prepare validation data
            validation_data = self._prepare_validation_data(data_df)
            
            # Test models on validation data
            tested_models = self._test_models_on_validation_data(selected_models, validation_data)
            
            if not tested_models:
                logger.warning("No models successfully tested, using fallback models")
                tested_models = self._generate_fallback_models()
            
            # Select best models
            best_models = self._select_best_models_from_validation(tested_models)
            
            # Store validation results
            self.validation_results = {
                'tested_models': tested_models,
                'best_models': best_models,
                'model_selection': model_selection
            }
            
            logger.info(f"Validation completed. Selected {len(best_models)} best models")
            return best_models
            
        except Exception as e:
            logger.error(f"Error in validation process: {e}")
            logger.info("Using fallback model selection")
            return self._generate_fallback_models()
    
    def _test_models_on_validation_data(self, selected_models: List[Dict], validation_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Test selected models on validation data"""
        tested_models = []
        
        for model_config in selected_models:
            model_name = model_config['model']
            hyperparameters = model_config['hyperparameters']
            
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Optimize hyperparameters for this model
                best_hyperparams, best_metrics = self._optimize_model_hyperparameters(
                    validation_data, model_name, hyperparameters
                )
                
                # Train final model with best hyperparameters
                final_model = self._train_final_model(validation_data, model_name, best_hyperparams)
                
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': best_hyperparams,
                    'validation_score': best_metrics['mae'],  # Using MAE as primary score
                    'validation_metrics': best_metrics,
                    'reason': model_config.get('reason', 'Optimized through validation'),
                    'trained_model': final_model
                })
                
                logger.info(f"Model {model_name} tested successfully with MAE: {best_metrics['mae']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to test model {model_name}: {e}")
                # Add with infinite score so it won't be selected
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': hyperparameters,
                    'validation_score': float('inf'),
                    'validation_metrics': {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')},
                    'reason': model_config.get('reason', 'Validation failed'),
                    'trained_model': None
                })
        
        return tested_models
    
    def _generate_fallback_models(self) -> List[Dict[str, Any]]:
        """Generate fallback models when validation fails"""
        logger.info("Generating fallback models...")
        
        fallback_models = []
        k_models = min(self.k_models, len(self.available_models))
        
        for i in range(k_models):
            model_name = self.available_models[i]
            fallback_models.append({
                'model': model_name,
                'hyperparameters': {},
                'validation_score': 1.0 + i * 0.1,
                'validation_metrics': {
                    'mse': 1.0 + i * 0.1,
                    'mae': 0.8 + i * 0.08,
                    'mape': 20.0 + i * 2.0
                },
                'reason': 'Fallback model selection',
                'trained_model': None
            })
        
        return fallback_models
    
    def _calculate_model_suitability_score(self, model: str, analysis_result: Dict[str, Any]) -> float:
        """Calculate model suitability score based on data characteristics"""
        score = 0.0
        
        # Extract data characteristics from analysis
        data_characteristics = analysis_result.get('data_characteristics', {})
        llm_insights = analysis_result.get('llm_insights', {})
        
        # Determine data properties from insights
        has_trend = 'trend' in str(llm_insights).lower()
        has_seasonality = 'seasonal' in str(llm_insights).lower() or 'period' in str(llm_insights).lower()
        is_stationary = 'stationary' in str(llm_insights).lower()
        
        # Score based on model characteristics and data characteristics
        if model == 'ARIMA' and is_stationary:
            score += 10
        elif model == 'ARIMA' and has_trend:
            score += 8
        elif model == 'SARIMA' and has_seasonality:
            score += 9
        elif model == 'Prophet' and (has_trend or has_seasonality):
            score += 8
        elif model in ['LSTM', 'VisionTransformer', 'SimpleVisionTransformer']:
            score += 7  # High versatility for complex patterns
        elif model in ['RandomForest', 'GradientBoosting']:
            score += 6  # Good for non-linear relationships
        elif model == 'LinearRegression' and has_trend:
            score += 5
        elif model == 'ExponentialSmoothing' and has_seasonality:
            score += 7
        elif model in ['BMTGNN', 'GraphTransformer', 'SimpleGraphTransformer', 'SpatioTemporalEnsemble', 'VisionTransformerForTimeSeries']:
            score += 8  # Advanced models for complex patterns
        else:
            score += 3  # Base score
        
        return score
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        report = {
            "validation_summary": {
                "total_tested": len(self.validation_results.get('tested_models', [])),
                "successfully_tested": len([m for m in self.validation_results.get('tested_models', []) 
                                          if m['validation_score'] < float('inf')]),
                "selected_models": len(self.validation_results.get('best_models', [])),
                "best_model": self.validation_results.get('best_models', [{}])[0].get('model') 
                             if self.validation_results.get('best_models') else None,
                "best_score": self.validation_results.get('best_models', [{}])[0].get('validation_score', float('inf'))
                             if self.validation_results.get('best_models') else float('inf')
            },
            "model_performance": {},
            "recommendations": []
        }
        
        # Add detailed performance for each model
        for model in self.validation_results.get('tested_models', []):
            report["model_performance"][model['model']] = {
                "validation_score": model['validation_score'],
                "metrics": model['validation_metrics'],
                "hyperparameters": model['hyperparameters']
            }
        
        # Generate recommendations
        best_models = self.validation_results.get('best_models', [])
        if best_models:
            report["recommendations"].append(f"Top performing model: {best_models[0]['model']} "
                                           f"(Score: {best_models[0]['validation_score']:.4f})")
            report["recommendations"].append(f"Recommended for ensemble: {[m['model'] for m in best_models]}")
        
        return report

    def select_models(self, analysis_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced model selection with validation capabilities"""
        logger.info("Starting model selection process...")
        
        data_df = self._ensure_dataframe(data)
        
        # Get initial model selection from LLM
        llm_selection = self._select_models_with_llm(analysis_results, data_df)
        
        # If validation data is available, perform hyperparameter optimization
        if self._has_validation_data(data_df):
            logger.info("Performing model validation and hyperparameter optimization...")
            validated_models = self._validate_and_optimize_models(llm_selection, data_df)
            return validated_models
        else:
            logger.info("Insufficient data for validation, using LLM selection only")
            return llm_selection
    
    def _select_models_with_llm(self, analysis_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Select best models using LLM guidance with structured output"""
        system_prompt = """You are a time series forecasting model selection expert specialising in cybersecurity data. 
        You analyse data characteristics and select the most appropriate forecasting models."""
        
        user_prompt = f"""
        Based on the following cybersecurity time series analysis:
        {json.dumps(analysis_results.get('llm_insights', {}), indent=2)}
        
        Data Information:
        - Shape: {data.shape}
        - Columns: {list(data.columns)}
        - Available models: {self.available_models}
        
        Select the top {self.n_candidates} most suitable models and recommend hyperparameters for each.
        
        Consider these data characteristics:
        - Stationarity (is the data stationary?)
        - Seasonality patterns (monthly, quarterly, yearly)
        - Trend characteristics (increasing, decreasing, stable)
        - Data frequency and volume
        - Cybersecurity context (potential attacks, anomalies, sparse events)
        
        For each model, provide:
        1. Model name (must be from available models)
        2. Hyperparameter search space (2-3 key parameters with reasonable ranges)
        3. Reasoning for selection
        
        Please give detailed and verbose reasoning for each choice.

        
        Return JSON format:
        {{
            "selected_models": [
                {{
                    "model": "str",
                    "hyperparameters": {{"param1": [value1, value2], "param2": [value1, value2]}},
                    "reason": "str"
                }}
            ],
            "ensemble_strategy": "str"
        }}
        """
        
        llm_response = self.llm_manager.generate_response(system_prompt, user_prompt)
        model_selection = self._parse_json_response(llm_response)
        
        # Validate and fallback if needed
        if not model_selection or "selected_models" not in model_selection:
            logger.warning("LLM model selection failed, using fallback models")
            model_selection = self._get_fallback_models()
        
        logger.info(f"Selected {len(model_selection['selected_models'])} models via LLM")
        return model_selection
    
    def _validate_and_optimize_models(self, model_selection: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Validate models and optimize hyperparameters"""
        logger.info("Starting model validation and hyperparameter optimization...")
        
        # Prepare validation data
        validation_data = self._prepare_validation_data(data)
        
        tested_models = []
        
        for model_config in model_selection["selected_models"]:
            model_name = model_config["model"]
            hyperparameters = model_config["hyperparameters"]
            
            try:
                logger.info(f"Validating and optimizing model: {model_name}")
                
                # Optimize hyperparameters for this model
                best_hyperparams, best_metrics = self._optimize_model_hyperparameters(
                    validation_data, model_name, hyperparameters
                )
                
                # Train final model with best hyperparameters on full validation data
                final_model = self._train_final_model(validation_data, model_name, best_hyperparams)
                
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': best_hyperparams,
                    'validation_score': best_metrics['mae'],  # Using MAE as primary score
                    'validation_metrics': best_metrics,
                    'reason': model_config.get('reason', 'Optimized through validation'),
                    'trained_model': final_model
                })
                
                logger.info(f"Model {model_name} optimized successfully with MAE: {best_metrics['mae']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to optimize model {model_name}: {e}")
                # Keep original configuration with poor validation score
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': hyperparameters,
                    'validation_score': float('inf'),
                    'validation_metrics': {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')},
                    'reason': model_config.get('reason', 'Validation failed'),
                    'trained_model': None
                })
        
        # Select best models based on validation performance
        best_models = self._select_best_models_from_validation(tested_models)
        
        # Store trained models
        for model_info in best_models:
            if model_info.get('trained_model'):
                model_name = model_info['model']
                self.trained_models[model_name] = model_info['trained_model']
        
        return {
            "selected_models": best_models,
            "ensemble_strategy": model_selection.get("ensemble_strategy", "weighted_average_based_on_metrics"),
            "validation_summary": {
                "total_tested": len(tested_models),
                "successfully_optimized": len([m for m in tested_models if m['validation_score'] < float('inf')]),
                "best_model": best_models[0]['model'] if best_models else None,
                "best_score": best_models[0]['validation_score'] if best_models else float('inf')
            }
        }
    
    def _prepare_validation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for validation"""
        # Use the last 20% of data for validation
        split_point = int(len(data) * 0.8)
        validation_data = data.iloc[split_point:].copy()
        
        # Ensure we have a 'value' column for compatibility
        if 'value' not in validation_data.columns and len(validation_data.columns) > 0:
            # Use first numeric column as value
            numeric_cols = validation_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                validation_data = validation_data.rename(columns={numeric_cols[0]: 'value'})
            else:
                # Create dummy value column if no numeric columns
                validation_data['value'] = range(len(validation_data))
        
        return validation_data
    
    def _has_validation_data(self, data: pd.DataFrame) -> bool:
        """Check if we have sufficient data for validation"""
        try:
            df = self._ensure_dataframe(data)
        except TypeError:
            return False
        return len(df) > 20  # Minimum data points for validation
    
    def _optimize_model_hyperparameters(self, validation_data: pd.DataFrame, 
                                      model_name: str, hyperparameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize hyperparameters for a specific model using grid search"""
        
        if not hyperparameters:
            logger.info(f"No hyperparameters to optimize for {model_name}, using defaults")
            default_metrics = self._evaluate_model_on_validation(validation_data, model_name, {})
            return {}, default_metrics
        
        # Generate parameter combinations (grid search)
        param_names = list(hyperparameters.keys())
        param_values = list(hyperparameters.values())
        
        # Limit the number of combinations to avoid excessive computation
        max_combinations = 20
        combinations = []
        
        # Generate combinations using itertools
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
            if len(combinations) >= max_combinations:
                break
        
        logger.info(f"Testing {len(combinations)} hyperparameter combinations for {model_name}...")
        
        best_params = {}
        best_metrics = {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}
        
        # Test each combination
        for i, params in enumerate(combinations):
            try:
                # Evaluate model with these parameters
                metrics = self._evaluate_model_on_validation(validation_data, model_name, params)
                
                if metrics['mae'] < best_metrics['mae']:
                    best_metrics = metrics
                    best_params = params.copy()
                
                logger.debug(f"Combination {i+1}/{len(combinations)}: MAE = {metrics['mae']:.4f}, Params = {params}")
                
            except Exception as e:
                logger.debug(f"Combination {i+1}/{len(combinations)} failed: {e}")
                continue
        
        logger.info(f"Best hyperparameters for {model_name}: {best_params} (MAE = {best_metrics['mae']:.4f})")
        return best_params, best_metrics
    
    def _evaluate_model_on_validation(self, validation_data: pd.DataFrame, 
                                    model_name: str, hyperparameters: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single model with specific hyperparameters on validation data"""
        try:
            # Split validation data into train/test
            split_point = int(len(validation_data) * 0.7)
            train_data = validation_data.iloc[:split_point]
            test_data = validation_data.iloc[split_point:]
            
            if len(train_data) == 0 or len(test_data) == 0:
                return {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}
            
            # Train model and get predictions
            model = self._train_model(train_data, model_name, hyperparameters)
            predictions = self._predict_with_model(model, test_data, model_name, hyperparameters)
            
            # Calculate metrics on the test portion
            actual_values = test_data['value'].values
            
            # Ensure predictions and actual values have same length
            if len(predictions) != len(actual_values):
                min_len = min(len(predictions), len(actual_values))
                predictions = predictions[:min_len]
                actual_values = actual_values[:min_len]
            
            if len(actual_values) == 0:
                return {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}
            
            # Calculate metrics
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            mape = np.mean(np.abs((actual_values - predictions) / np.where(actual_values != 0, actual_values, 1))) * 100
            rae = np.sum(np.abs(predictions - actual_values)) / np.sum(np.abs(actual_values - np.mean(actual_values)))
            rrse = np.sqrt(np.sum((predictions - actual_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))
            
            # Calculate R² score
            if np.var(actual_values) > 0:
                r2 = r2_score(actual_values, predictions)
            else:
                r2 = -float('inf')
            
            return {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'rae': rae,
                'rrse': rrse,
                'r2': r2
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'rae': float('inf'),
                'rrse': float('inf'),
                'r2': -float('inf')
            }
    
    def _validate_time_series_data(self, ts_data: Dict[str, pd.Series], node_names: List[str]):
        """
        Validate the prepared time series data.
        
        Args:
            ts_data: Dictionary of time series data
            node_names: Expected node names
        """
        logger.info("Validating time series data...")
        
        validation_results = {
            'total_nodes': len(ts_data),
            'nodes_with_data': 0,
            'nodes_missing': [],
            'data_lengths': {},
            'data_stats': {}
        }
        
        for node in node_names:
            if node in ts_data:
                series = ts_data[node]
                
                # Check for basic validity
                if len(series) > 0 and not series.isna().all():
                    validation_results['nodes_with_data'] += 1
                    validation_results['data_lengths'][node] = len(series)
                    
                    # Calculate basic statistics
                    validation_results['data_stats'][node] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max(),
                        'na_count': series.isna().sum()
                    }
                    
                    logger.debug(f"Node {node}: {len(series)} points, mean={series.mean():.2f}")
                else:
                    validation_results['nodes_missing'].append(node)
                    logger.warning(f"Node {node} has no valid data")
            else:
                validation_results['nodes_missing'].append(node)
                logger.warning(f"Node {node} not found in time series data")
        
        # Log validation summary
        logger.info(f"Time series data validation: {validation_results['nodes_with_data']}/{len(node_names)} nodes have valid data")
        
        if validation_results['nodes_missing']:
            logger.warning(f"Missing data for nodes: {validation_results['nodes_missing']}")
        
        return validation_results

    # change the signature and docstring near prepare_time_series_data
    def prepare_time_series_data(self, data: Union[pd.DataFrame, Dict[str, Any], Any], node_names: List[str]) -> Dict[str, pd.Series]:
        """
        Prepare time series data for forecasting by converting DataFrame or dict to dictionary of Series.
        
        Args:
            data: DataFrame with time series data (rows: time, columns: nodes) or a dict keyed by node name
            node_names: List of node/column names
        Returns: Dictionary mapping node names to pandas Series
        
        """
        logger.info("Preparing time series data for forecasting...")
        
        ts_data = {}
        
        # If data is already in the right format with node names as columns
        if isinstance(data, pd.DataFrame):
            for node in node_names:
                if node in data.columns:
                    # Create time series for this node
                    ts_data[node] = data[node].copy()
                    
                    # Ensure the series has a proper time index
                    if not isinstance(ts_data[node].index, pd.DatetimeIndex):
                        # Create a monthly time index (common for cybersecurity data)
                        start_date = pd.Timestamp('2011-07-01')  # Common start for cybersecurity data
                        dates = pd.date_range(
                            start=start_date, 
                            periods=len(ts_data[node]), 
                            freq='M'
                        )
                        ts_data[node].index = dates
                    
                    logger.debug(f"Prepared time series for {node}: {len(ts_data[node])} points")
                else:
                    logger.warning(f"Node {node} not found in data columns")
        
        # If data is a dictionary or needs transformation
        elif isinstance(data, dict):
            for node in node_names:
                if node in data:
                    ts_data[node] = pd.Series(data[node])
                    # Add time index
                    start_date = pd.Timestamp('2011-07-01')
                    dates = pd.date_range(
                        start=start_date, 
                        periods=len(ts_data[node]), 
                        freq='M'
                    )
                    ts_data[node].index = dates
                else:
                    logger.warning(f"Node {node} not found in data dictionary")
        
        else:
            # Fallback: create synthetic data for each node
            logger.warning("Data format not recognized, creating synthetic time series")
            for i, node in enumerate(node_names):
                # Create synthetic time series data
                n_points = len(data) if hasattr(data, '__len__') else 100
                synthetic_data = np.random.randn(n_points).cumsum() + 100  # Random walk
                ts_data[node] = pd.Series(synthetic_data)
                
                # Add time index
                start_date = pd.Timestamp('2011-07-01')
                dates = pd.date_range(
                    start=start_date, 
                    periods=len(ts_data[node]), 
                    freq='M'
                )
                ts_data[node].index = dates
        
        # Validate the prepared data
        self._validate_time_series_data(ts_data, node_names)
        
        logger.info(f"Successfully prepared time series data for {len(ts_data)} nodes")
        return ts_data


    def _train_model(self, train_data: pd.DataFrame, model_name: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train one or more models for each node and collect performance."""
        trained_models: Dict[str, Dict[str, Any]] = {}
        model_performance: Dict[str, Dict[str, Any]] = {}
        
        try:
            
            train_size = int(len(series) * self.config.validation_split)
            train_data = series[:train_size]
            test_data = series[train_size:]
            # Prepare per-node time series data
            ts_data = self.prepare_time_series_data(train_data, list(train_data.columns))

            for node, series in ts_data.items():
                logger.info(f"Training models for node: {node}")

                # Train/validation split
                train_size = int(len(series) * self.config.validation_split)
                train_data = series[:train_size]
                test_data = series[train_size:]
                
                node_train = series[:train_size]
                node_val = series[train_size:]

                node_models: Dict[str, Any] = {}
                node_metrics: Dict[str, Any] = {}

                # Loop over selected models
                for model_config in model_selection["selected_models"]:

                    model_name = model_config["model"]
                    hyperparameters = model_config.get("hyperparameters", hyperparameters)

                    try:
                        if model_name == "ARIMA":
                            return self._train_arima(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "SARIMA":
                            return self._train_sarima(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "ExponentialSmoothing":
                            return self._train_exponential_smoothing(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "RandomForest":
                            return self._train_random_forest(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "LinearRegression":
                            return self._train_linear_regression(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "SVR":
                            return self._train_svr(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "GradientBoosting":
                            return self._train_gradient_boosting(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "LSTM":
                            return self._train_lstm(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "Prophet":
                            return self._train_prophet(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "BMTGNN":
                            return self._train_bmtgnn(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "VisionTransformer":
                            return self._train_vision_transformer(train_data, hyperparameters)
                        elif model_name == "VisionTransformer":
                            return self._train_vision_transformer(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "SimpleVisionTransformer":
                            model, performance = self._train_simple_vision_transformer(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "SpatioTemporalEnsemble" and ENSEMBLE_AVAILABLE:
                            return self._train_spatiotemporal_ensemble(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "VisionTransformerForTimeSeries" and ENSEMBLE_AVAILABLE:
                            return self._train_vision_transformer_timeseries(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "SimpleGraphTransformer":
                            return self._train_simple_graph_transformer(train_data, test_data, model_config["hyperparameters"])
                        elif model_name == "GraphTransformer":
                            return self._train_graph_transformer(train_data, test_data, model_config["hyperparameters"])
                        else:
                            continue

                        # If here, training succeeded for this model/node
                        node_models[model_name] = model
                        node_metrics[model_name] = performance

                    except Exception as e:
                        logger.error(f"Error training {model_name} for node {node}: {e}")

                # After all models for this node are attempted, store results
                trained_models[node] = node_models
                model_performance[node] = node_metrics

        except Exception as e:
            logger.error(f"Training failed in _train_model for '{model_name}': {e}")

        # Safely return even if some nodes/models failed
        return {
            "trained_models": trained_models,
            "performance": model_performance
        }

    def _predict_with_model(self, model: Any, test_data: pd.DataFrame, model_name: str, 
                          hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Generate predictions with a trained model"""
        try:
            test_values = test_data['value'].values
            horizon = len(test_values)
            
            if model_name == "ARIMA":
                return self._predict_arima(model, horizon)
            elif model_name == "SARIMA":
                return self._predict_sarima(model, horizon)
            elif model_name == "ExponentialSmoothing":
                return self._predict_exponential_smoothing(model, horizon)
            elif model_name == "RandomForest":
                return self._predict_random_forest(model, test_values, hyperparameters)
            elif model_name == "LinearRegression":
                return self._predict_linear_regression(model, test_values, hyperparameters)
            elif model_name == "SVR":
                return self._predict_svr(model, test_values, hyperparameters)
            elif model_name == "GradientBoosting":
                return self._predict_gradient_boosting(model, test_values, hyperparameters)
            elif model_name == "LSTM":
                return self._predict_lstm(model, test_values, hyperparameters)
            elif model_name == "Prophet":
                return self._predict_prophet(model, test_data, horizon)
            elif model_name == "BMTGNN":
                return self._predict_bmtgnn(model, test_data, horizon)
            elif model_name == "VisionTransformer":
                return self._predict_vision_transformer(model, test_data, horizon)
            elif model_name == "SimpleVisionTransformer":
                return self._predict_simple_vision_transformer(model, test_data, horizon)
            elif model_name == "SpatioTemporalEnsemble" and ENSEMBLE_AVAILABLE:
                return self._predict_spatiotemporal_ensemble(model, test_data, horizon)
            elif model_name == "VisionTransformerForTimeSeries" and ENSEMBLE_AVAILABLE:
                return self._predict_vision_transformer_timeseries(model, test_data, horizon)
            elif model_name == "SimpleGraphTransformer":
                return self._predict_simple_graph_transformer(model, test_data, horizon)
            elif model_name == "GraphTransformer":
                return self._predict_graph_transformer(model, test_data, horizon)
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            # Return simple fallback predictions
            return np.full(horizon, np.mean(test_values) if len(test_values) > 0 else 0)
    
    def _train_final_model(self, data: pd.DataFrame, model_name: str, hyperparameters: Dict[str, Any]) -> Any:
        """Train final model on full data with best hyperparameters"""
        return self._train_model(data, model_name, hyperparameters)
    
    # INDIVIDUAL MODEL IMPLEMENTATIONS
    
    def _train_arima(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train ARIMA model"""
        p = hyperparameters.get('p', 1)
        d = hyperparameters.get('d', 1)
        q = hyperparameters.get('q', 1)
        
        try:
            model = ARIMA(values, order=(p, d, q))
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # Fallback to simple configuration
            model = ARIMA(values, order=(1, 1, 1))
            return model.fit()
    
    def _predict_arima(self, model: Any, horizon: int) -> np.ndarray:
        """Predict with ARIMA model"""
        try:
            forecast = model.forecast(steps=horizon)
            return forecast.values
        except:
            # Fallback prediction
            return np.full(horizon, model.model.endog[-1])
    
    def _train_sarima(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train SARIMA model"""
        p = hyperparameters.get('p', 1)
        d = hyperparameters.get('d', 1)
        q = hyperparameters.get('q', 1)
        P = hyperparameters.get('P', 1)
        D = hyperparameters.get('D', 1)
        Q = hyperparameters.get('Q', 1)
        s = hyperparameters.get('s', 12)
        
        try:
            model = SARIMAX(values, order=(p, d, q), seasonal_order=(P, D, Q, s))
            fitted_model = model.fit(disp=False)
            return fitted_model
        except Exception as e:
            # Fallback to simple configuration
            model = SARIMAX(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            return model.fit(disp=False)
    
    def _predict_sarima(self, model: Any, horizon: int) -> np.ndarray:
        """Predict with SARIMA model"""
        try:
            forecast = model.forecast(steps=horizon)
            return forecast.values
        except:
            return np.full(horizon, model.model.endog[-1])
    
    def _train_exponential_smoothing(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train Exponential Smoothing model"""
        trend = hyperparameters.get('trend', 'add')
        seasonal = hyperparameters.get('seasonal', 'add')
        seasonal_periods = hyperparameters.get('seasonal_periods', 12)
        
        try:
            model = ExponentialSmoothing(
                values, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # Fallback to simple configuration
            model = ExponentialSmoothing(values, trend='add', seasonal=None)
            return model.fit()
    
    def _predict_exponential_smoothing(self, model: Any, horizon: int) -> np.ndarray:
        """Predict with Exponential Smoothing model"""
        try:
            forecast = model.forecast(horizon)
            return forecast.values
        except:
            return np.full(horizon, model.fittedvalues[-1])
    
    def _train_random_forest(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train Random Forest model"""
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', 10)
        min_samples_split = hyperparameters.get('min_samples_split', 2)
        
        # Create features for time series
        X, y = self._create_time_series_features(values, lag=5)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for Random Forest")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _predict_random_forest(self, model: Any, test_values: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with Random Forest model"""
        if len(test_values) == 0:
            return np.array([])
        
        last_values = test_values[-5:] if len(test_values) >= 5 else test_values
        if len(last_values) < 5:
            last_values = np.pad(last_values, (5 - len(last_values), 0), mode='edge')
        
        X_test = np.array([last_values] * len(test_values))
        predictions = model.predict(X_test)
        
        return predictions[:len(test_values)]
    
    def _train_linear_regression(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train Linear Regression model"""
        fit_intercept = hyperparameters.get('fit_intercept', True)
        
        # Create features for time series
        X, y = self._create_time_series_features(values, lag=3)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for Linear Regression")
        
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)
        return model
    
    def _predict_linear_regression(self, model: Any, test_values: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with Linear Regression model"""
        # Similar to Random Forest, create features for test period
        if len(test_values) == 0:
            return np.array([])
        
        last_values = test_values[-3:] if len(test_values) >= 3 else test_values
        if len(last_values) < 3:
            last_values = np.pad(last_values, (3 - len(last_values), 0), mode='edge')
        
        X_test = np.array([last_values] * len(test_values))
        predictions = model.predict(X_test)
        
        return predictions[:len(test_values)]
    
    def _train_svr(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train Support Vector Regression model"""
        C = hyperparameters.get('C', 1.0)
        kernel = hyperparameters.get('kernel', 'rbf')
        gamma = hyperparameters.get('gamma', 'scale')
        
        X, y = self._create_time_series_features(values, lag=4)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for SVR")
        
        model = SVR(C=C, kernel=kernel, gamma=gamma)
        model.fit(X, y)
        return model
    
    def _predict_svr(self, model: Any, test_values: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with SVR model"""
        if len(test_values) == 0:
            return np.array([])
        
        last_values = test_values[-4:] if len(test_values) >= 4 else test_values
        if len(last_values) < 4:
            last_values = np.pad(last_values, (4 - len(last_values), 0), mode='edge')
        
        X_test = np.array([last_values] * len(test_values))
        predictions = model.predict(X_test)
        
        return predictions[:len(test_values)]
    
    def _train_gradient_boosting(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train Gradient Boosting model"""
        n_estimators = hyperparameters.get('n_estimators', 100)
        learning_rate = hyperparameters.get('learning_rate', 0.1)
        max_depth = hyperparameters.get('max_depth', 3)
        
        X, y = self._create_time_series_features(values, lag=4)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for Gradient Boosting")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _predict_gradient_boosting(self, model: Any, test_values: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with Gradient Boosting model"""
        if len(test_values) == 0:
            return np.array([])
        
        last_values = test_values[-4:] if len(test_values) >= 4 else test_values
        if len(last_values) < 4:
            last_values = np.pad(last_values, (4 - len(last_values), 0), mode='edge')
        
        X_test = np.array([last_values] * len(test_values))
        predictions = model.predict(X_test)
        
        return predictions[:len(test_values)]
    
    def _train_lstm(self, values: np.ndarray, hyperparameters: Dict[str, Any]) -> Any:
        """Train LSTM model"""
        hidden_dim = hyperparameters.get('hidden_dim', 50)
        num_layers = hyperparameters.get('num_layers', 2)
        sequence_length = hyperparameters.get('sequence_length', 12)
        epochs = hyperparameters.get('epochs', 50)
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        
        # Normalize data
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(values_scaled)):
            X.append(values_scaled[i-sequence_length:i])
            y.append(values_scaled[i])
        
        if len(X) == 0:
            raise ValueError("Insufficient data for LSTM")
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
        y_tensor = torch.FloatTensor(y)
        
        # Create model
        model = LSTMModel(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        # Store scaler for inverse transformation
        self.scalers['LSTM'] = scaler
        return model
    
    def _predict_lstm(self, model: Any, test_values: np.ndarray, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with LSTM model"""
        if len(test_values) == 0:
            return np.array([])
        
        sequence_length = hyperparameters.get('sequence_length', 12)
        scaler = self.scalers.get('LSTM')
        
        if scaler is None:
            # Create a new scaler if not found
            scaler = MinMaxScaler()
            test_values_scaled = scaler.fit_transform(test_values.reshape(-1, 1)).flatten()
        else:
            test_values_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
        
        # Prepare input sequence
        if len(test_values_scaled) < sequence_length:
            # Pad if insufficient data
            padded = np.pad(test_values_scaled, (sequence_length - len(test_values_scaled), 0), mode='edge')
            input_seq = padded
        else:
            input_seq = test_values_scaled[-sequence_length:]
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            current_seq = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
            
            for _ in range(len(test_values)):
                pred = model(current_seq)
                predictions.append(pred.item())
                
                # Update sequence
                new_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)], dim=1)
                current_seq = new_seq
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()
        
        return predictions[:len(test_values)]
    
    def _train_prophet(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Any:
        """Train Prophet model"""
        yearly_seasonality = hyperparameters.get('yearly_seasonality', True)
        weekly_seasonality = hyperparameters.get('weekly_seasonality', False)
        daily_seasonality = hyperparameters.get('daily_seasonality', False)
        changepoint_prior_scale = hyperparameters.get('changepoint_prior_scale', 0.05)
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': data.index if isinstance(data.index, pd.DatetimeIndex) else pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
            'y': data['value'].values
        })
        
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        model.fit(prophet_df)
        return model
    
    def _predict_prophet(self, model: Any, test_data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict with Prophet model"""
        # Create future dataframe
        future_dates = pd.date_range(
            start=test_data.index[-1] + pd.Timedelta(days=1) if isinstance(test_data.index, pd.DatetimeIndex) else pd.Timestamp('2023-01-01'),
            periods=horizon,
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        return forecast['yhat'].values
    
    def _train_vision_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Vision Transformer model """
        try:
            from hyperoptim_transfer_learning_rework import VisionTransformerForTimeSeries
            
            # Prepare configuration 
            config = {
                'sequence_length': hyperparams.get('sequence_length', 24),
                'patch_size': hyperparams.get('patch_size', 4),
                'embed_dim': hyperparams.get('embed_dim', 128),
                'num_heads': hyperparams.get('num_heads', 8),
                'hidden_dim': hyperparams.get('hidden_dim', 256),
                'num_layers': hyperparams.get('num_layers', 3),
                'dropout': hyperparams.get('dropout', 0.1),
                'mc_dropout': hyperparams.get('mc_dropout', 0.2),
                'batch_size': hyperparams.get('batch_size', 8),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'learning_rate': hyperparams.get('learning_rate', 0.001)
            }
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = VisionTransformerForTimeSeries(config).to(device)
            
            # Prepare data for Vision Transformer
            train_tensor = torch.FloatTensor(train_data.values).unsqueeze(-1).unsqueeze(0)  # (1, seq_len, 1)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(50):  
                optimizer.zero_grad()
                output = model(train_tensor)
                loss = criterion(output.squeeze(), torch.FloatTensor([train_data.iloc[-1]]))  # Simplified target
                loss.backward()
                optimizer.step()
            
            # Generate forecast
            model.eval()
            with torch.no_grad():
                input_seq = torch.FloatTensor(train_data.values[-config['sequence_length']:]).unsqueeze(0).unsqueeze(-1)
                forecast = model(input_seq).squeeze().numpy()
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return model
            
        except Exception as e:
            logger.error(f"VisionTransformer training error: {e}")
            # Return fallback
            return None, {}

    def _train_simple_vision_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Simple Vision Transformer model"""
        try:
            from hyperparams_optim_vanilla_vit_transfer_learning import SimpleVisionTransformer
            
            config = {
                'sequence_length': hyperparams.get('sequence_length', 12),
                'patch_size': hyperparams.get('patch_size', 4),
                'embed_dim': hyperparams.get('embed_dim', 128),
                'num_heads': hyperparams.get('num_heads', 8),
                'hidden_dim': hyperparams.get('hidden_dim', 256),
                'num_layers': hyperparams.get('num_layers', 3),
                'dropout': hyperparams.get('dropout', 0.1),
                'batch_size': hyperparams.get('batch_size', 8),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'learning_rate': hyperparams.get('learning_rate', 0.001)
            }
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleVisionTransformer(config).to(device)
            
            # Prepare data for Simple Vision Transformer
            train_tensor = torch.FloatTensor(train_data.values).unsqueeze(-1).unsqueeze(0)  # (1, seq_len, 1)
            
            # Simple training loop 
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(50): 
                optimizer.zero_grad()
                output = model(train_tensor)
                loss = criterion(output.squeeze(), torch.FloatTensor([train_data.iloc[-1]]))  # Simplified target
                loss.backward()
                optimizer.step()
            
            # Generate forecast
            model.eval()
            with torch.no_grad():
                input_seq = torch.FloatTensor(train_data.values[-config['sequence_length']:]).unsqueeze(0).unsqueeze(-1)
                forecast = model(input_seq).squeeze().numpy()
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return model
            
        except Exception as e:
            logger.error(f"SimpleVisionTransformer training error: {e}")
            # Return fallback
            return None, {}
        
    def _train_bmtgnn(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Bayesian MTGNN model"""
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare data for B-MTGNN (convert to multi-variate format expected by the model)
            # Since B-MTGNN expects multi-variate data, we'll create a single-variable version
            data_values = train_data.values.reshape(-1, 1)  # Convert to 2D array with 1 feature
            
            # Create a temporary CSV file with the training data
            temp_data_path = "temp_bmtgnn_data.csv"
            pd.DataFrame(data_values).to_csv(temp_data_path, index=False, header=False)
            
            # Initialize B-MTGNN data loader with the same parameters 
            bmtgnn_data = DataLoaderS(
                file_name=temp_data_path,
                train=0.7,
                valid=0.15,
                device=device,
                horizon=1,
                window=hyperparams.get('seq_in_len', 6),
                normalize=2,
                out=hyperparams.get('seq_out_len', 12)
            )
            
            # Extract hyperparameters with defaults 
            gcn_depth = hyperparams.get('gcn_depth', 2)
            conv_channels = hyperparams.get('conv_channels', 8)
            residual_channels = hyperparams.get('residual_channels', 32)
            skip_channels = hyperparams.get('skip_channels', 128)
            end_channels = hyperparams.get('end_channels', 256)
            layers = hyperparams.get('layers', 2)
            subgraph_size = hyperparams.get('subgraph_size', 20)
            dropout = hyperparams.get('dropout', 0.3)
            dilation_exponential = hyperparams.get('dilation_exponential', 2)
            node_dim = hyperparams.get('node_dim', 40)
            propalpha = hyperparams.get('propalpha', 0.05)
            tanhalpha = hyperparams.get('tanhalpha', 3)
            
            # Initialize B-MTGNN model with exact architecture
            model = gtnet(
                gcn_true=True,
                buildA_true=True,
                gcn_depth=gcn_depth,
                num_nodes=1,  # Single node for univariate series
                device=device,
                predefined_A=None,
                dropout=dropout,
                subgraph_size=subgraph_size,
                node_dim=node_dim,
                dilation_exponential=dilation_exponential,
                conv_channels=conv_channels,
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                end_channels=end_channels,
                seq_length=hyperparams.get('seq_in_len', 6),
                in_dim=1,
                out_dim=hyperparams.get('seq_out_len', 12),
                layers=layers,
                propalpha=propalpha,
                tanhalpha=tanhalpha,
                layer_norm_affline=False
            )
            
            # Set random seed for reproducibility
            set_random_seed(123)
            
            # Initialize optimizer
            learning_rate = hyperparams.get('learning_rate', 0.001)
            optim = Optim(model.parameters(), 'adam', learning_rate, clip=5)
            
            # Training loop 
            model.train()
            epochs = hyperparams.get('epochs', 20)
            batch_size = hyperparams.get('batch_size', 4)
            
            for epoch in range(epochs):
                train_loss = 0
                n_samples = 0
                
                for X_batch, Y_batch in bmtgnn_data.get_batches(
                    bmtgnn_data.train[0], bmtgnn_data.train[1], batch_size, True
                ):
                    model.zero_grad()
                    X_batch = torch.unsqueeze(X_batch, dim=1)
                    X_batch = X_batch.transpose(2, 3)
                    
                    output = model(X_batch)
                    output = torch.squeeze(output, 3)
                    
                    # Apply scaling
                    scale = bmtgnn_data.scale.expand(output.size(0), output.size(1), bmtgnn_data.m)
                    output = output * scale
                    Y_batch = Y_batch * scale
                    
                    # Calculate loss
                    criterion = torch.nn.L1Loss(reduction='sum')
                    loss = criterion(output, Y_batch)
                    loss.backward()
                    
                    train_loss += loss.item()
                    n_samples += (output.size(0) * output.size(1) * bmtgnn_data.m)
                    optim.step()
            
            # Generate predictions for evaluation
            model.eval()
            with torch.no_grad():
                # Use the test window approach 
                test_window = bmtgnn_data.test_window
                if test_window is not None and len(test_window) > 0:
                    # Simple prediction using the trained model
                    X_test = torch.unsqueeze(test_window[:hyperparams.get('seq_in_len', 6)], dim=0)
                    X_test = torch.unsqueeze(X_test, dim=1)
                    X_test = X_test.transpose(2, 3)
                    
                    predictions = []
                    for _ in range(10):  # Multiple runs for Bayesian estimation
                        output = model(X_test)
                        predictions.append(output.squeeze().cpu().numpy())
                    
                    # Use mean prediction
                    forecast = np.mean(predictions, axis=0)
                    
                    # Ensure forecast length matches test data
                    if len(forecast) > len(test_data):
                        forecast = forecast[:len(test_data)]
                    elif len(forecast) < len(test_data):
                        forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
                else:
                    # Fallback: use simple forecast
                    forecast = np.full(len(test_data), train_data.iloc[-1])
            
            # Calculate metrics
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model
            
        except Exception as e:
            logger.error(f"BMTGNN training error: {e}")
            # Clean up temporary file if it exists
            import os
            temp_data_path = "temp_bmtgnn_data.csv"
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            return None, {}

    def _predict_bmtgnn(self, model: Any, test_data: pd.Series, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with BMTGNN model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # Prepare input data
            seq_in_len = hyperparameters.get('seq_in_len', 6)
            input_seq = test_data.values[-seq_in_len:]
            input_tensor = torch.FloatTensor(input_seq).reshape(1, 1, 1, -1)  # (batch, channels, nodes, seq_len)
            input_tensor = input_tensor.transpose(2, 3)  # Adjust dimensions for B-MTGNN
            
            # Generate predictions with multiple runs for Bayesian estimation
            predictions = []
            with torch.no_grad():
                for _ in range(10):  # Multiple runs
                    output = model(input_tensor)
                    pred = output.squeeze().cpu().numpy()
                    predictions.append(pred)
            
            # Use mean prediction
            forecast = np.mean(predictions, axis=0)
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return forecast
            
        except Exception as e:
            logger.error(f"BMTGNN prediction error: {e}")
            return np.full(len(test_data), np.mean(test_data.values) if len(test_data) > 0 else 0)
        
    def _train_simple_graph_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train SimpleGraphTransformer model"""
        try:
            # Import the required classes
            from pretrain_script import SimpleGraphTransformer, SimpleDataLoader
            
            # Prepare configuration
            config = {
                'channels': hyperparams.get('channels', 64),
                'num_layers': hyperparams.get('num_layers', 3),
                'pe_dim': hyperparams.get('pe_dim', 8),
                'pe_walk_length': hyperparams.get('pe_walk_length', 20),
                'node_dim': hyperparams.get('node_dim', 1),
                'num_heads': hyperparams.get('num_heads', 4),
                'attn_dropout': hyperparams.get('attn_dropout', 0.2),
                'dropout': hyperparams.get('dropout', 0.3),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'sequence_length': hyperparams.get('sequence_length', 12)
            }
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleGraphTransformer(config).to(device)
            
            # Prepare data in the format expected by SimpleDataLoader
            # Create a temporary CSV file with the training data
            temp_data_path = "temp_simple_gt_data.csv"
            data_df = pd.DataFrame({'value': train_data.values})
            data_df.to_csv(temp_data_path, index=False)
            
            # Create data loader
            data_loader = SimpleDataLoader(
                temp_data_path,
                device,
                seq_len=config['sequence_length'],
                out_len=len(test_data),
                correlation_threshold=hyperparams.get('correlation_threshold', 0.3)
            )
            
            # Simple training loop 
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.get('learning_rate', 0.001))
            
            model.train()
            epochs = hyperparams.get('epochs', 50)
            batch_size = hyperparams.get('batch_size', 16)
            
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
                
                # Prepare PyG data
                train_pyg_data = data_loader.prepare_pyg_data(
                    data_loader.train[0].numpy(), 
                    data_loader.train[1].numpy()
                )
                
                if len(train_pyg_data) == 0:
                    continue
                    
                train_loader = PyGDataLoader(train_pyg_data, batch_size=min(batch_size, len(train_pyg_data)), shuffle=True)
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    # Reshape target to match output
                    batch_size_out = output.shape[0]
                    target_reshaped = batch.y.view(batch_size_out, data_loader.out_len, data_loader.m)
                    
                    loss = criterion(output, target_reshaped)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
            
            # Generate predictions
            model.eval()
            with torch.no_grad():
                # Use the last sequence from training data
                if len(data_loader.train[0]) > 0:
                    last_sequence = data_loader.train[0][-1].unsqueeze(0).numpy()
                    last_target = data_loader.train[1][-1].unsqueeze(0).numpy()
                    
                    test_pyg_data = data_loader.prepare_pyg_data(last_sequence, last_target)
                    if len(test_pyg_data) > 0:
                        test_batch = test_pyg_data[0].to(device)
                        forecast = model(
                            test_batch.x.unsqueeze(0), 
                            test_batch.pe.unsqueeze(0), 
                            test_batch.edge_index, 
                            test_batch.edge_attr, 
                            torch.tensor([0])
                        )
                        forecast = forecast.squeeze().cpu().numpy()
                    else:
                        forecast = np.full(len(test_data), train_data.iloc[-1])
                else:
                    forecast = np.full(len(test_data), train_data.iloc[-1])
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            # Calculate metrics
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model
            
        except Exception as e:
            logger.error(f"SimpleGraphTransformer training error: {e}")
            # Clean up temporary file if it exists
            import os
            temp_data_path = "temp_simple_gt_data.csv"
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            return None, {}
        
    def _predict_simple_graph_transformer(self, model: Any, test_data: pd.Series, horizon: int) -> np.ndarray:
        """Predict with SimpleGraphTransformer model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # Create a simple edge index (self-connections for single node)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
            edge_attr = torch.tensor([1], dtype=torch.long).to(device)
            
            # Use the last value as input
            last_value = test_data.iloc[-1] if len(test_data) > 0 else 0
            
            # Create positional encoding
            pe_walk_length = 20  # Default from config
            pe = torch.randn(1, pe_walk_length).to(device)  # PE
            
            # Prepare batch
            batch = torch.tensor([0], dtype=torch.long).to(device)
            
            # Generate predictions step by step
            predictions = []
            current_input = torch.FloatTensor([[last_value]]).to(device)
            
            with torch.no_grad():
                for _ in range(horizon):
                    output = model(current_input, pe, edge_index, edge_attr, batch)
                    pred_value = output.item() if output.numel() == 1 else output[0, 0, 0].item()
                    predictions.append(pred_value)
                    
                    # Update input for next step (use prediction as next input)
                    current_input = torch.FloatTensor([[pred_value]]).to(device)
            
            predictions = np.array(predictions)
            
            # Clip predictions to reasonable range
            if len(test_data) > 0:
                data_min = test_data.min()
                data_max = test_data.max()
                predictions = np.clip(predictions, data_min * 0.5, data_max * 1.5)
            
            return predictions
            
        except Exception as e:
            logger.error(f"SimpleGraphTransformer prediction error: {e}")
            # Fallback: use simple moving average
            if len(test_data) > 0:
                window_size = min(3, len(test_data))
                moving_avg = test_data.rolling(window=window_size).mean().iloc[-1]
                if pd.isna(moving_avg):
                    moving_avg = test_data.mean()
                return np.full(horizon, moving_avg)
            else:
                return np.zeros(horizon)

    def _train_graph_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train GraphTransformer model"""
        try:
            # Import the required classes
            from transfer_learning_hyperparams_optim import GraphTransformer, CyberThreatDataLoader
            
            # Prepare configuration
            config = {
                'channels': hyperparams.get('channels', 64),
                'num_layers': hyperparams.get('num_layers', 3),
                'pe_dim': hyperparams.get('pe_dim', 8),
                'pe_walk_length': hyperparams.get('pe_walk_length', 20),
                'node_dim': hyperparams.get('node_dim', 1),
                'num_heads': hyperparams.get('num_heads', 4),
                'attn_dropout': hyperparams.get('attn_dropout', 0.2),
                'dropout': hyperparams.get('dropout', 0.3),
                'weight_decay': hyperparams.get('weight_decay', 1e-4),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'sequence_length': hyperparams.get('sequence_length', 18),
                'local_gnn_type': hyperparams.get('local_gnn_type', 'GINE'),
                'norm_type': hyperparams.get('norm_type', 'batch'),
                'head_hidden_dims': hyperparams.get('head_hidden_dims', [128, 64]),
                'head_activation': hyperparams.get('head_activation', 'relu')
            }
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GraphTransformer(config).to(device)
            
            # Prepare data in the format expected by CyberThreatDataLoader
            temp_data_path = "temp_gt_data.csv"
            data_df = pd.DataFrame({'value': train_data.values})
            data_df.to_csv(temp_data_path, index=False)
            
            # Create data loader
            data_loader = CyberThreatDataLoader(
                temp_data_path,
                device,
                seq_len=config['sequence_length'],
                out_len=len(test_data),
                correlation_threshold=hyperparams.get('correlation_threshold', 0.3),
                max_edges_per_node=hyperparams.get('max_edges_per_node', 15)
            )
            
            # Training setup
            criterion = nn.MSELoss()
            learning_rate = hyperparams.get('learning_rate', 0.0005)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
            
            model.train()
            epochs = hyperparams.get('epochs', 50)
            batch_size = hyperparams.get('batch_size', 8)
            
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
                
                # Prepare PyG data
                train_pyg_data = data_loader.prepare_pyg_data(
                    data_loader.train[0].numpy(), 
                    data_loader.train[1].numpy(),
                    walk_length=config['pe_walk_length']
                )
                
                if len(train_pyg_data) == 0:
                    continue
                    
                train_loader = PyGDataLoader(train_pyg_data, batch_size=min(batch_size, len(train_pyg_data)), shuffle=True)
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    # Reshape target to match output
                    batch_size_out = output.shape[0]
                    target_reshaped = batch.y.view(batch_size_out, data_loader.train_horizon, data_loader.m)
                    
                    loss = criterion(output, target_reshaped)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams.get('grad_clip', 1.0))
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
            
            # Generate predictions
            model.eval()
            with torch.no_grad():
                if len(data_loader.train[0]) > 0:
                    last_sequence = data_loader.train[0][-1].unsqueeze(0).numpy()
                    last_target = data_loader.train[1][-1].unsqueeze(0).numpy()
                    
                    test_pyg_data = data_loader.prepare_pyg_data(last_sequence, last_target)
                    if len(test_pyg_data) > 0:
                        test_batch = test_pyg_data[0].to(device)
                        forecast = model(
                            test_batch.x.unsqueeze(0), 
                            test_batch.pe.unsqueeze(0), 
                            test_batch.edge_index, 
                            test_batch.edge_attr, 
                            torch.tensor([0])
                        )
                        forecast = forecast.squeeze().cpu().numpy()
                    else:
                        forecast = np.full(len(test_data), train_data.iloc[-1])
                else:
                    forecast = np.full(len(test_data), train_data.iloc[-1])
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            # Calculate metrics
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model
            
        except Exception as e:
            logger.error(f"GraphTransformer training error: {e}")
            # Clean up temporary file if it exists
            import os
            temp_data_path = "temp_gt_data.csv"
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            return None, {}
        
    def _predict_graph_transformer(self, model: Any, test_data: pd.Series, horizon: int) -> np.ndarray:
        """Predict with GraphTransformer model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()

            # Create simple graph structure for single node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
            edge_attr = torch.tensor([1], dtype=torch.long).to(device)
            
            # Get model configuration
            channels = model.channels if hasattr(model, 'channels') else 64
            pe_walk_length = model.pe_walk_length if hasattr(model, 'pe_walk_length') else 20
            
            # Prepare positional encoding
            pe = torch.randn(1, pe_walk_length).to(device)
            
            # Prepare batch
            batch = torch.tensor([0], dtype=torch.long).to(device)
            
            # Use Monte Carlo dropout for uncertainty estimation
            mc_samples = 10
            all_predictions = []
            
            # Use the last few values as context
            context_size = min(12, len(test_data))
            if context_size == 0:
                last_value = 0
            else:
                last_value = test_data.iloc[-1]
            
            current_input = torch.FloatTensor([[last_value]]).to(device)
            
            # Generate predictions with MC dropout
            with torch.no_grad():
                for _ in range(mc_samples):
                    step_predictions = []
                    step_input = current_input.clone()
                    
                    for _ in range(horizon):
                        # Enable dropout for MC sampling
                        model.train()
                        output = model(step_input, pe, edge_index, edge_attr, batch, 
                                    mc_dropout=True, forecast_horizon=1)
                        
                        # Extract prediction
                        if output.dim() == 3:
                            pred_value = output[0, 0, 0].item()
                        else:
                            pred_value = output.item()
                        
                        step_predictions.append(pred_value)
                        
                        # Update input for next step
                        step_input = torch.FloatTensor([[pred_value]]).to(device)
                    
                    all_predictions.append(step_predictions)
            
            # Average predictions across MC samples
            all_predictions = np.array(all_predictions)
            predictions = np.mean(all_predictions, axis=0)
            
            # Clip to reasonable range
            if len(test_data) > 0:
                data_mean = test_data.mean()
                data_std = test_data.std()
                lower_bound = data_mean - 3 * data_std
                upper_bound = data_mean + 3 * data_std
                predictions = np.clip(predictions, lower_bound, upper_bound)
            
            return predictions
            
        except Exception as e:
            logger.error(f"GraphTransformer prediction error: {e}")
            # Fallback: exponential smoothing
            if len(test_data) > 0:
                # Simple exponential smoothing
                alpha = 0.3
                smoothed = []
                last = test_data.iloc[0]
                for val in test_data:
                    last = alpha * val + (1 - alpha) * last
                    smoothed.append(last)
                
                # Extend the trend
                if len(smoothed) >= 2:
                    trend = smoothed[-1] - smoothed[-2]
                    forecast = [smoothed[-1] + trend * (i+1) for i in range(horizon)]
                else:
                    forecast = [smoothed[-1]] * horizon if smoothed else [test_data.mean()] * horizon
                
                return np.array(forecast)
            else:
                return np.zeros(horizon)
    
    def _predict_vision_transformer(self, model: Any, test_data: pd.Series, horizon: int) -> np.ndarray:
        """
        Generate predictions using the VisionTransformer model.
        
        Args:
            model: Trained VisionTransformerForTimeSeries model
            test_data: Test data series
            horizon: Forecast horizon
            
        Returns:
            Array of predictions
        """
        try:
            logger.info(f"Generating VisionTransformer predictions for horizon: {horizon}")
            
            # Get model configuration
            config = model.config if hasattr(model, 'config') else {}
            sequence_length = config.get('sequence_length', 24)
            
            # Prepare input sequence
            if len(test_data) >= sequence_length:
                # Use last sequence_length points from test data
                input_seq = test_data.values[-sequence_length:]
            else:
                # Pad if insufficient data
                logger.warning(f"Insufficient test data ({len(test_data)} < {sequence_length}), padding")
                input_seq = np.pad(test_data.values, 
                                  (sequence_length - len(test_data), 0), 
                                  mode='edge')
            
            # Convert to tensor format: (batch_size=1, seq_len, num_nodes=1)
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
            
            # Set model to evaluation mode
            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                output = model(input_tensor, mc_dropout=False)  # Disable MC dropout for deterministic predictions
                predictions = output.squeeze().cpu().numpy()
            
            # Ensure we have the right number of predictions
            if len(predictions) > horizon:
                predictions = predictions[:horizon]
            elif len(predictions) < horizon:
                # Pad if model outputs fewer predictions than needed
                logger.warning(f"Model output ({len(predictions)}) less than horizon ({horizon}), padding")
                predictions = np.pad(predictions, 
                                    (0, horizon - len(predictions)), 
                                    mode='edge')
            
            # Apply post-processing if needed
            predictions = self._post_process_predictions(predictions, test_data)
            
            logger.info(f"VisionTransformer predictions generated: {len(predictions)} points")
            return predictions
            
        except Exception as e:
            logger.error(f"VisionTransformer prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback: use mean of test data
            fallback_value = np.mean(test_data.values) if len(test_data) > 0 else 0
            return np.full(horizon, fallback_value)

    def _predict_simple_vision_transformer(self, model: Any, test_data: pd.Series, horizon: int) -> np.ndarray:
        """
        Generate predictions using the SimpleVisionTransformer model.
        
        Args:
            model: Trained SimpleVisionTransformer model
            test_data: Test data series
            horizon: Forecast horizon
            
        Returns:
            Array of predictions
        """
        try:
            logger.info(f"Generating SimpleVisionTransformer predictions for horizon: {horizon}")
            
            # Get model configuration
            config = model.config if hasattr(model, 'config') else {}
            sequence_length = config.get('sequence_length', 12)
            
            # Prepare input sequence
            if len(test_data) >= sequence_length:
                # Use last sequence_length points from test data
                input_seq = test_data.values[-sequence_length:]
            else:
                # Pad if insufficient data
                logger.warning(f"Insufficient test data ({len(test_data)} < {sequence_length}), padding")
                input_seq = np.pad(test_data.values, 
                                  (sequence_length - len(test_data), 0), 
                                  mode='edge')
            
            # Convert to tensor format: (batch_size=1, seq_len, num_nodes=1)
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
            
            # Set model to evaluation mode
            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                output = model(input_tensor)
                predictions = output.squeeze().cpu().numpy()
            
            # Ensure we have the right number of predictions
            if len(predictions) > horizon:
                predictions = predictions[:horizon]
            elif len(predictions) < horizon:
                # Pad if model outputs fewer predictions than needed
                logger.warning(f"Model output ({len(predictions)}) less than horizon ({horizon}), padding")
                predictions = np.pad(predictions, 
                                    (0, horizon - len(predictions)), 
                                    mode='edge')
            
            # Apply post-processing if needed
            predictions = self._post_process_predictions(predictions, test_data)
            
            logger.info(f"SimpleVisionTransformer predictions generated: {len(predictions)} points")
            return predictions
            
        except Exception as e:
            logger.error(f"SimpleVisionTransformer prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback: use mean of test data
            fallback_value = np.mean(test_data.values) if len(test_data) > 0 else 0
            return np.full(horizon, fallback_value)

    def _post_process_predictions(self, predictions: np.ndarray, test_data: pd.Series) -> np.ndarray:
        """
        Apply post-processing to predictions.
        
        Args:
            predictions: Raw model predictions
            test_data: Original test data for context
            
        Returns:
            Post-processed predictions
        """
        try:
            # 1. Clip extreme values
            if len(test_data) > 0:
                # Use test data statistics for clipping
                test_mean = np.mean(test_data.values)
                test_std = np.std(test_data.values)
                
                # Define reasonable bounds (mean ± 3*std)
                lower_bound = test_mean - 3 * test_std
                upper_bound = test_mean + 3 * test_std
                
                predictions = np.clip(predictions, lower_bound, upper_bound)
            
            # 2. Smooth predictions if they're too volatile
            if len(predictions) > 3:
                # Apply simple moving average smoothing
                window_size = min(3, len(predictions))
                smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
                
                # Blend original and smoothed (70% original, 30% smoothed)
                predictions = 0.7 * predictions + 0.3 * smoothed
            
            # 3. Ensure non-negative predictions for count-like data
            if np.all(test_data.values >= 0):
                predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return predictions

    def _train_spatiotemporal_ensemble(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train SpatioTemporalEnsemble model using exact architecture from ensemble script"""
        try:
            # Prepare configuration for SpatioTemporalEnsemble
            config = {
                'sequence_length': hyperparams.get('sequence_length', 12),
                'vit_patch_size': hyperparams.get('vit_patch_size', 4),
                'vit_embed_dim': hyperparams.get('vit_embed_dim', 128),
                'vit_num_heads': hyperparams.get('vit_num_heads', 8),
                'vit_hidden_dim': hyperparams.get('vit_hidden_dim', 256),
                'vit_num_layers': hyperparams.get('vit_num_layers', 3),
                'graph_hidden_dim': hyperparams.get('graph_hidden_dim', 128),
                'graph_num_layers': hyperparams.get('graph_num_layers', 2),
                'fusion_dim': hyperparams.get('fusion_dim', 256),
                'dropout': hyperparams.get('dropout', 0.2),
                'mc_dropout': hyperparams.get('mc_dropout', 0.2),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'batch_size': hyperparams.get('batch_size', 8),
                'learning_rate': hyperparams.get('learning_rate', 0.001)
            }
            
            model.config = config

            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SpatioTemporalEnsemble(config).to(device)
            
            # Prepare data - convert to the format expected by CyberThreatDataset
            train_values = train_data.values.reshape(-1, 1)  # Convert to 2D with 1 feature
            test_values = test_data.values.reshape(-1, 1)
            
            # Create dataset
            dataset = CyberThreatDataset(
                data=train_values,
                sequence_length=config['sequence_length'],
                forecast_horizon=config['forecast_horizon']
            )
            
            # Simple training loop 
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(50): 
                total_loss = 0
                for i in range(len(dataset)):
                    if i >= len(dataset):
                        break
                        
                    X, y = dataset[i]
                    X = X.unsqueeze(0).to(device)  # Add batch dimension
                    y = y.unsqueeze(0).to(device)
                    
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Generate forecast
            model.eval()
            with torch.no_grad():
                # Use last sequence for prediction
                if len(train_values) >= config['sequence_length']:
                    input_seq = train_values[-config['sequence_length']:]
                    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)  # (1, seq_len, 1)
                    forecast = model(input_tensor).squeeze().cpu().numpy()
                else:
                    # Fallback if insufficient data
                    forecast = np.full(len(test_data), train_data.iloc[-1])
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return model
            
        except Exception as e:
            logger.error(f"SpatioTemporalEnsemble training error: {e}")
            return None, {}

    def _train_vision_transformer_timeseries(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train VisionTransformerForTimeSeries model """
        try:
            # Prepare configuration for VisionTransformerForTimeSeries
            config = {
                'sequence_length': hyperparams.get('sequence_length', 12),
                'patch_size': hyperparams.get('patch_size', 4),
                'embed_dim': hyperparams.get('embed_dim', 128),
                'num_heads': hyperparams.get('num_heads', 8),
                'hidden_dim': hyperparams.get('hidden_dim', 256),
                'num_layers': hyperparams.get('num_layers', 3),
                'dropout': hyperparams.get('dropout', 0.2),
                'mc_dropout': hyperparams.get('mc_dropout', 0.2),
                'forecast_horizon': len(test_data),
                'num_nodes': 1,  # Single node for univariate series
                'batch_size': hyperparams.get('batch_size', 8),
                'learning_rate': hyperparams.get('learning_rate', 0.001)
            }
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = VisionTransformerForTimeSeries(config).to(device)
            
            # Prepare data
            train_values = train_data.values.reshape(-1, 1)
            
            # Create dataset
            dataset = CyberThreatDataset(
                data=train_values,
                sequence_length=config['sequence_length'],
                forecast_horizon=config['forecast_horizon']
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(50):
                total_loss = 0
                for i in range(min(len(dataset), 10)): 
                    X, y = dataset[i]
                    X = X.unsqueeze(0).to(device)
                    y = y.unsqueeze(0).to(device)
                    
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Generate forecast
            model.eval()
            with torch.no_grad():
                if len(train_values) >= config['sequence_length']:
                    input_seq = train_values[-config['sequence_length']:]
                    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
                    forecast = model(input_tensor).squeeze().cpu().numpy()
                else:
                    forecast = np.full(len(test_data), train_data.iloc[-1])
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return model
            
        except Exception as e:
            logger.error(f"VisionTransformerForTimeSeries training error: {e}")
            return None, {}
        
    def _predict_spatiotemporal_ensemble(self, model: Any, test_data: pd.Series, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with SpatioTemporalEnsemble model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # Get model configuration
            config = model.config
            sequence_length = config['sequence_length']
            forecast_horizon = len(test_data)
            
            # Prepare input sequence
            test_values = test_data.values
            if len(test_values) >= sequence_length:
                # Use the last 'sequence_length' values from test data for prediction
                input_seq = test_values[-sequence_length:]
            else:
                # Pad if insufficient data
                input_seq = np.pad(test_values, (sequence_length - len(test_values), 0), mode='edge')
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(input_seq).reshape(1, sequence_length, 1).to(device)
            
            # Generate predictions
            predictions = []
            with torch.no_grad():
                # For Bayesian estimation with MC dropout
                num_runs = 10
                outputs = []
                
                for run in range(num_runs):
                    # Enable MC dropout for uncertainty estimation
                    output = model(input_tensor, mc_dropout=True)
                    output = output.squeeze().cpu().numpy()
                    
                    # Ensure output matches forecast horizon
                    if len(output) > forecast_horizon:
                        output = output[:forecast_horizon]
                    elif len(output) < forecast_horizon:
                        output = np.pad(output, (0, forecast_horizon - len(output)), mode='edge')
                    
                    outputs.append(output)
                
                # Use mean prediction across runs
                outputs_array = np.array(outputs)
                predictions = np.mean(outputs_array, axis=0)
                
                # Calculate confidence intervals
                confidence = 1.96 * np.std(outputs_array, axis=0) / np.sqrt(num_runs)
                
                # Can store confidence intervals if needed
                model.confidence_intervals = confidence
            
            return predictions
            
        except Exception as e:
            logger.error(f"SpatioTemporalEnsemble prediction error: {e}")
            # Fallback prediction
            if len(test_data) > 0:
                return np.full(len(test_data), np.mean(test_data.values))
            else:
                return np.array([])

    def _predict_vision_transformer_timeseries(self, model: Any, test_data: pd.Series, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Predict with VisionTransformerForTimeSeries model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # Get model configuration
            config = model.config
            sequence_length = config['sequence_length']
            forecast_horizon = len(test_data)
            
            # Prepare input sequence
            test_values = test_data.values
            if len(test_values) >= sequence_length:
                input_seq = test_values[-sequence_length:]
            else:
                input_seq = np.pad(test_values, (sequence_length - len(test_values), 0), mode='edge')
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(input_seq).reshape(1, sequence_length, 1).to(device)
            
            # Generate predictions with MC dropout for uncertainty
            predictions = []
            with torch.no_grad():
                num_runs = 10
                outputs = []
                
                for run in range(num_runs):
                    # Enable MC dropout
                    output = model(input_tensor, mc_dropout=True)
                    output = output.squeeze().cpu().numpy()
                    
                    # Adjust output length
                    if len(output) > forecast_horizon:
                        output = output[:forecast_horizon]
                    elif len(output) < forecast_horizon:
                        output = np.pad(output, (0, forecast_horizon - len(output)), mode='edge')
                    
                    outputs.append(output)
                
                # Use mean prediction
                outputs_array = np.array(outputs)
                predictions = np.mean(outputs_array, axis=0)
                
                # Store confidence intervals
                confidence = 1.96 * np.std(outputs_array, axis=0) / np.sqrt(num_runs)
                model.confidence_intervals = confidence
            
            return predictions
            
        except Exception as e:
            logger.error(f"VisionTransformerForTimeSeries prediction error: {e}")
            # Fallback prediction
            if len(test_data) > 0:
                return np.full(len(test_data), np.mean(test_data.values))
            else:
                return np.array([])

    def _predict_spatiotemporal_ensemble_simple(self, model: Any, test_data: pd.Series, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """ Prediction with SpatioTemporalEnsemble """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            config = model.config
            sequence_length = config['sequence_length']
            
            test_values = test_data.values
            if len(test_values) >= sequence_length:
                input_seq = test_values[-sequence_length:]
            else:
                input_seq = np.pad(test_values, (sequence_length - len(test_values), 0), mode='edge')
            
            input_tensor = torch.FloatTensor(input_seq).reshape(1, sequence_length, 1).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, mc_dropout=False)
                forecast = output.squeeze().cpu().numpy()
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return forecast
            
        except Exception as e:
            logger.error(f"Simple SpatioTemporalEnsemble prediction error: {e}")
            return np.full(len(test_data), np.mean(test_data.values) if len(test_data) > 0 else 0)

    def _predict_vision_transformer_timeseries_simple(self, model: Any, test_data: pd.Series, hyperparameters: Dict[str, Any]) -> np.ndarray:
        """Prediction with VisionTransformerForTimeSeries"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            config = model.config
            sequence_length = config['sequence_length']
            
            test_values = test_data.values
            if len(test_values) >= sequence_length:
                input_seq = test_values[-sequence_length:]
            else:
                input_seq = np.pad(test_values, (sequence_length - len(test_values), 0), mode='edge')
            
            input_tensor = torch.FloatTensor(input_seq).reshape(1, sequence_length, 1).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, mc_dropout=False)
                forecast = output.squeeze().cpu().numpy()
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            return forecast
            
        except Exception as e:
            logger.error(f"Simple VisionTransformerForTimeSeries prediction error: {e}")
            return np.full(len(test_data), np.mean(test_data.values) if len(test_data) > 0 else 0)

    def _create_time_series_features(self, values: np.ndarray, lag: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for time series models"""
        X, y = [], []
        
        for i in range(lag, len(values)):
            X.append(values[i-lag:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def _select_best_models_from_validation(self, tested_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the best models based on validation performance"""
        if not tested_models:
            return []
        
        # Filter out models that failed validation
        valid_models = [m for m in tested_models if m['validation_score'] < float('inf')]
        
        if not valid_models:
            logger.warning("No models passed validation, using first few models")
            return tested_models[:self.k_models]
        
        # Sort by validation score (lower is better)
        sorted_models = sorted(valid_models, key=lambda x: x['validation_score'])
        
        # Select top k_models
        best_models = sorted_models[:self.k_models]
        
        logger.info(f"Selected best models based on validation:")
        for model in best_models:
            metrics = model['validation_metrics']
            logger.info(f"  {model['model']}: MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
        
        return best_models
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
        
        # Fallback to basic model selection
        return self._get_fallback_models()
    
    def _get_fallback_models(self) -> Dict[str, Any]:
        """Provide fallback model selection including ALL available models"""
        return {
            "selected_models": [
                {
                    "model": "ARIMA",
                    "hyperparameters": {"p": [0,1,2], "d": [0,1], "q": [0,1,2]},
                    "reason": "General purpose time series model for stationary data"
                },
                {
                    "model": "ExponentialSmoothing",
                    "hyperparameters": {
                        "trend": ["add", "mul"], 
                        "seasonal": ["add", "mul"],
                        "seasonal_periods": [12]
                    },
                    "reason": "Handles trend and seasonality in cybersecurity patterns"
                },
                {
                    "model": "RandomForest",
                    "hyperparameters": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, 15],
                        "min_samples_split": [2, 5]
                    },
                    "reason": "Robust to outliers and anomalies in threat data"
                },
                {
                    "model": "LinearRegression",
                    "hyperparameters": {
                        "fit_intercept": [True, False]
                    },
                    "reason": "Baseline linear model for benchmarking"
                },
                {
                    "model": "LSTM",
                    "hyperparameters": {
                        "hidden_dim": [32, 50, 64],
                        "num_layers": [1, 2, 3],
                        "sequence_length": [6, 12, 24],
                        "epochs": [50, 100],
                        "learning_rate": [0.001, 0.01]
                    },
                    "reason": "Deep learning for complex temporal patterns in cyber threats"
                },
                {
                    "model": "Prophet",
                    "hyperparameters": {
                        "yearly_seasonality": [True, False],
                        "weekly_seasonality": [False],
                        "daily_seasonality": [False],
                        "changepoint_prior_scale": [0.01, 0.05, 0.1]
                    },
                    "reason": "Facebook Prophet for automatic seasonality detection"
                },
                {
                "model": "SpatioTemporalEnsemble",
                "hyperparameters": {
                    "sequence_length": [12, 24, 36],
                    "vit_patch_size": [2, 3, 4, 6],
                    "vit_embed_dim": [64, 96, 128, 160],
                    "vit_num_heads": [4, 8],
                    "vit_hidden_dim": [128, 192, 256, 320],
                    "vit_num_layers": [2, 3, 4],
                    "graph_hidden_dim": [64, 96, 128, 160],
                    "graph_num_layers": [1, 2, 3],
                    "fusion_dim": [128, 192, 256, 320],
                    "dropout": [0.1, 0.2, 0.3],
                    "mc_dropout": [0.1, 0.2, 0.3, 0.4],
                    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.002],
                    "weight_decay": [0.001, 0.01, 0.05],
                    "batch_size": [4, 8, 16]
                },
                "reason": "Advanced spatio-temporal ensemble combining Vision Transformer and graph-aware branches with cross-attention fusion"
            },
                {
                    "model": "VisionTransformer",
                    "hyperparameters": {
                        "sequence_length": [12, 24, 36],
                        "patch_size": [2, 4, 6],
                        "embed_dim": [64, 128, 256],
                        "num_heads": [4, 8],
                        "hidden_dim": [128, 256, 512],
                        "num_layers": [2, 3, 4],
                        "dropout": [0.1, 0.2, 0.3],
                        "learning_rate": [0.0001, 0.0005, 0.001],
                        "batch_size": [8, 16, 32]
                    },
                    "reason": "Advanced Vision Transformer architecture for time series with transfer learning capabilities"
                },
                {
                    "model": "SimpleVisionTransformer", 
                    "hyperparameters": {
                        "sequence_length": [12, 24],
                        "patch_size": [2, 4, 6],
                        "embed_dim": [64, 128],
                        "num_heads": [4, 8],
                        "hidden_dim": [128, 256],
                        "num_layers": [2, 3],
                        "dropout": [0.1, 0.2],
                        "learning_rate": [0.0001, 0.001],
                        "batch_size": [8, 16]
                    },
                    "reason": "Vision Transformer optimized for time series forecasting with robust patch embedding"
                    },
                
                {
                "model": "VisionTransformerForTimeSeries",
                "hyperparameters": {
                    "sequence_length": [12, 24, 36],
                    "patch_size": [2, 3, 4, 6],
                    "embed_dim": [64, 96, 128, 160],
                    "num_heads": [4, 8],
                    "hidden_dim": [128, 192, 256, 320],
                    "num_layers": [2, 3, 4],
                    "dropout": [0.1, 0.2, 0.3],
                    "mc_dropout": [0.1, 0.2, 0.3, 0.4],
                    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.002],
                    "weight_decay": [0.001, 0.01, 0.05],
                    "batch_size": [4, 8, 16]
                },
                "reason": "Pure Vision Transformer architecture for time series with patch embedding and transformer encoder"
                },
                
                {"model": "BMTGNN",
                "hyperparameters": {
                    "gcn_depth": [1, 2, 3],
                    "conv_channels": [4, 8, 16],
                    "residual_channels": [16, 32, 64],
                    "skip_channels": [64, 128, 256],
                    "end_channels": [256, 512, 1024],
                    "layers": [1, 2],
                    "subgraph_size": [20, 30],
                    "dropout": [0.2, 0.3, 0.4],
                    "dilation_exponential": [1, 2],
                    "node_dim": [40, 50, 60],
                    "propalpha": [0.05, 0.1, 0.2],
                    "tanhalpha": [1, 2, 3],
                    "learning_rate": [0.001, 0.0005, 0.0001]
                },
                "reason": "Bayesian MTGNN for graph-based time series forecasting with uncertainty estimation"
                },
                 {
                "model": "SimpleGraphTransformer",
                "hyperparameters": {
                    "channels": [32, 64, 96],
                    "num_layers": [2, 3, 4],
                    "pe_dim": [4, 8, 12],
                    "pe_walk_length": [10, 15, 20],
                    "node_dim": [1],
                    "num_heads": [4, 6, 8],
                    "attn_dropout": [0.1, 0.15, 0.2],
                    "dropout": [0.1, 0.2, 0.3],
                    "forecast_horizon": [6, 12, 24],
                    "sequence_length": [12, 18, 24],
                    "correlation_threshold": [0.2, 0.3, 0.4],
                    "local_gnn_type": ["GINE", "GCN"],
                    "learning_rate": [0.0001, 0.0005, 0.001],
                    "batch_size": [8, 16, 32]
                },
                "reason": "Graph Transformer with GPS convolution and positional encoding for spatio-temporal cybersecurity data"
            },
                {   
                "model": "GraphTransformer",
                "hyperparameters": {
                    "channels": [32, 64, 96, 128],
                    "num_layers": [2, 3, 4],
                    "pe_dim": [4, 8, 12],
                    "pe_walk_length": [20],
                    "node_dim": [1],
                    "num_heads": [4, 6, 8],
                    "attn_dropout": [0.1, 0.15, 0.2],
                    "dropout": [0.1, 0.2, 0.3],
                    "weight_decay": [1e-5, 5e-5, 1e-4],
                    "learning_rate": [5e-5, 1e-4, 2.5e-4, 5e-4],
                    "batch_size": [4, 8, 12],
                    "forecast_horizon": [36],
                    "sequence_length": [12, 15, 18],
                    "correlation_threshold": [0.25, 0.3, 0.35],
                    "max_edges_per_node": [10, 15],
                    "local_gnn_type": ["GINE", "GCN"],
                    "norm_type": ["batch", "layer"],
                    "head_hidden_dims": [[128, 64], [256, 128]],
                    "head_activation": ["relu", "gelu"],
                    "grad_clip": [0.5, 1.0, 2.0]
                },
                "reason": "Advanced Graph Transformer with comprehensive hyperparameter optimization and transfer learning capabilities"
                }
            ],
            "ensemble_strategy": "weighted_average_based_on_metrics"}
    
    def get_trained_model(self, model_name: str) -> Any:
        """Get a trained model by name"""
        return self.trained_models.get(model_name)
    
    def save_models(self, directory: str):
        """Save trained models to directory"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            try:
                if model_name == "LSTM":
                    # Save PyTorch model
                    torch.save(model.state_dict(), Path(directory) / f"{model_name}_model.pth")
                else:
                    # Save other models with joblib
                    joblib.dump(model, Path(directory) / f"{model_name}_model.joblib")
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, Path(directory) / "scalers.joblib")
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models from directory"""
        for model_name in self.trained_models.keys():
            try:
                if model_name == "LSTM":
                    # Load PyTorch model
                    model_path = Path(directory) / f"{model_name}_model.pth"
                    if model_path.exists():
                        # Recreate model architecture first
                        hyperparams = self._get_default_lstm_hyperparams()
                        model = LSTMModel(
                            input_dim=1,
                            hidden_dim=hyperparams['hidden_dim'],
                            num_layers=hyperparams['num_layers'],
                            output_dim=1
                        )
                        model.load_state_dict(torch.load(model_path))
                        self.trained_models[model_name] = model
                else:
                    # Load other models
                    model_path = Path(directory) / f"{model_name}_model.joblib"
                    if model_path.exists():
                        self.trained_models[model_name] = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        # Load scalers
        scalers_path = Path(directory) / "scalers.joblib"
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
        
        logger.info(f"Models loaded from {directory}")
    
    def _get_default_lstm_hyperparams(self) -> Dict[str, Any]:
        """Get default LSTM hyperparameters"""
        return {
            'hidden_dim': 50,
            'num_layers': 2,
            'sequence_length': 12,
            'epochs': 50,
            'learning_rate': 0.001
        }