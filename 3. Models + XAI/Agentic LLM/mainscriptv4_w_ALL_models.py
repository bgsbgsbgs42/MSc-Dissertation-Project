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
from dataclasses import dataclass, field
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import openai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pgmpy.estimators import PC, HillClimbSearch, BIC
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import scipy.stats as stats
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict
from ModelSelectionAgentII import ModelSelectionAgent
from AnalysisAgent import AnalysisAgent
from LSTMForecaster import LSTMForecaster
from RAGSystem import RAGSystem
from XAIAnalyzer import XAIAnalyzer
from ReportAgent import ReportAgent
from VisualizationAgent import VisualizationAgent
from PATRecommendationAgent import PATRecommendationAgent, PATConfig
from preprocess_agent import PreprocessAgent
from torch_geometric.loader import DataLoader as PyGDataLoader

try:
    from hyperparameter_optim_bmtgnn_transfer_learning import (
        gtnet, DataLoaderS, Optim, set_random_seed
    )
    BMTGNN_AVAILABLE = True
except ImportError:
    BMTGNN_AVAILABLE = False
    print("B-MTGNN not available - ensure hyperparameter_optim_bmtgnn_transfer_learning.py is accessible")

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
    from pretrain_script import SimpleGraphTransformer, SimpleDataLoader
    from transfer_learning_hyperparams_optim import GraphTransformer, CyberThreatDataLoader
    GRAPH_TRANSFORMERS_AVAILABLE = True
except ImportError:
    GRAPH_TRANSFORMERS_AVAILABLE = False
    print("Graph Transformer models not available - ensure pretrain_script.py and transfer_learning_hyperparams_optim.py are accessible")


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ForecastConfig:
    """Configuration for the forecasting system"""
    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "deepseek-r1:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    
    # Data Configuration
    data_path: str = "data/sm_data_g.csv"
    graph_path: str = "data/graph.csv"
    forecast_months: int = 36
    target_columns: List[str] = None
    
    # Model Configuration
    validation_split: float = 0.8
    seasonal_period: int = 12
    
    # Output Configuration
    output_dir: str = "forecast_results"
    plot_style: str = "default"
    
    # Agent Configuration
    enable_agents: bool = True
    max_retries: int = 3
    
    # XAI Configuration
    enable_xai: bool = True
    xai_methods: List[str] = None
    
    # RAG Configuration
    enable_rag: bool = False
    rag_db_path: str = "cyber_threat_db"

    # Report Configuration
    enab_report: bool = True
    report_path: str = "forecast_report.pdf"    
    
    # New Models
    enable_lstm: bool = True
    enable_prophet: bool = True
    enable_validation: bool = True  
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
            'n_candidates': 3,
            'k_models': 3, 
            'cv_folds': 5,
            'optimization_method': 'grid_search'
    })
    
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
            print(f"LLM initialized successfully with model: {self.config.llm_model}")
            return Ollama(
                model=self.config.llm_model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=0.1
            )
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
            if hasattr(response, 'content'):
                return response.content
            return str(response)
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
        
        # Load graph metadata for node names - which become the target columns
        graph_data = pd.read_csv(graph_path)
        target_columns = graph_data.columns.tolist()  # COLUMNS FROM graph.csv ARE TARGETS
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
        3. Data normalisation
        4. Feature engineering for cybersecurity context
        
        Please give highly detailed and verbose reasoning for each choice.

        
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

        
class EnhancedForecastingEngine:
    """Enhanced forecasting engine with XAI, RAG, and additional models"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.data_agent = DataAgent(self.llm_manager)
        self.analysis_agent = AnalysisAgent(self.llm_manager)
        self.model_agent = ModelSelectionAgent(self.llm_manager)
        self.xai_analyzer = XAIAnalyzer(config)

         # Initialize Report Agent
        self.report_agent = ReportAgent(
            llm_manager=self.llm_manager,
            config={'output_dir': Path(config.output_dir) / "reports"}
        )
        

        if config.enable_prophet:
            self.model_agent.available_models.append("Prophet")
        if BMTGNN_AVAILABLE:
            self.model_agent.available_models.append("BMTGNN")
        if ENSEMBLE_AVAILABLE:
            self.model_agent.available_models.extend(["SpatioTemporalEnsemble", "VisionTransformerForTimeSeries"])
                
        self.model_agent.available_models.extend(["VisionTransformer", "SimpleVisionTransformer"])
        self.model_agent.available_models.extend(["SimpleGraphTransformer", "GraphTransformer"])

    def prepare_time_series_data(self, data: pd.DataFrame, node_names: List[str]) -> Dict[str, pd.Series]:
        """
        Prepare time series data for forecasting by converting DataFrame to dictionary of Series.
        
        Args:
            data: DataFrame with time series data (rows: time, columns: nodes)
            node_names: List of node/column names
            
        Returns:
            Dictionary mapping node names to pandas Series
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

    def _get_available_models(self, config: ForecastConfig) -> List[str]:
        """Get list of available models based on configuration"""
        base_models = [
            "ARIMA", "ExponentialSmoothing", "RandomForest", "LinearRegression", 
            "SARIMA", "SVR", "GradientBoosting"
        ]
        
        if config.enable_lstm:
            base_models.append("LSTM")
        if config.enable_prophet:
            base_models.append("Prophet")
        if BMTGNN_AVAILABLE:
            base_models.append("BMTGNN")
        if ENSEMBLE_AVAILABLE:
            base_models.extend(["SpatioTemporalEnsemble", "VisionTransformerForTimeSeries"])
                
        base_models.extend(["VisionTransformer", "SimpleVisionTransformer"])
        
        if GRAPH_TRANSFORMERS_AVAILABLE:
            base_models.extend(["SimpleGraphTransformer", "GraphTransformer"])
            
        return base_models

    def run_comprehensive_validation(self, analysis_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation matching ValidationAgent functionality"""
        logger.info("Running comprehensive model validation...")

        #  handle dict/DF inputs gracefully 
        data_df = data
        node_names = None
        if isinstance(data, dict):
            data_df = data["data"] if "data" in data and data["data"] is not None else data.get("cleaned_data")
            node_names = data.get("node_names")

        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("run_comprehensive_validation expects a DataFrame or dict containing a 'data' DataFrame")

        if node_names is None:
            node_names = list(data_df.columns)

        ts_data = self.prepare_time_series_data(data_df, node_names)
        
        # Use the enhanced validation method
        best_models = self.model_agent.run_validation(analysis_results, data)
        
        # Generate validation report
        validation_report = self.model_agent.generate_validation_report()
        
        # Prepare time series data for training
        ts_data = self.prepare_time_series_data(data_df, list(data_df.columns))
        
        # Train the selected models
        training_results = self.train_models(ts_data, {"selected_models": best_models})
        
        return {
            "validation_results": validation_report,
            "best_models": best_models,
            "training_results": training_results,
            "trained_models": self.model_agent.trained_models
        }
    
    def _get_available_models(self, config: ForecastConfig) -> List[str]:
        """Get list of available models based on configuration"""
        base_models = [
            "ARIMA", "ExponentialSmoothing", "RandomForest", "LinearRegression", 
            "SARIMA", "SVR", "GradientBoosting"
        ]
        
        if config.enable_lstm:
            base_models.append("LSTM")
        if config.enable_prophet:
            base_models.append("Prophet")
        if BMTGNN_AVAILABLE:
            base_models.append("BMTGNN")
        if ENSEMBLE_AVAILABLE:
            base_models.extend(["SpatioTemporalEnsemble", "VisionTransformerForTimeSeries"])
                
        base_models.extend(["VisionTransformer", "SimpleVisionTransformer"])
        
        if GRAPH_TRANSFORMERS_AVAILABLE:
            base_models.extend(["SimpleGraphTransformer", "GraphTransformer"])
            
        return base_models
    
    def prepare_time_series_data(self, data: pd.DataFrame, node_names: List[str]) -> Dict[str, pd.Series]:
        """
        Prepare time series data for forecasting by converting DataFrame to dictionary of Series.
        
        Args:
            data: DataFrame with time series data (rows: time, columns: nodes)
            node_names: List of node/column names
            
        Returns:
            Dictionary mapping node names to pandas Series
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
    
    def train_models(self, ts_data: Dict[str, pd.Series], model_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced model training"""
        trained_models = {}
        model_performance = {}
        
        for node, series in list(ts_data.items())[:1]:  # Train on first node for demo
            logger.info(f"Training models for node: {node}")
            
            # Prepare train/test split
            train_size = int(len(series) * self.config.validation_split)
            train_data = series[:train_size]
            test_data = series[train_size:]
            
            node_models = {}
            node_performance = {}
            
            for model_config in model_selection["selected_models"]:
                model_name = model_config["model"]
                try:
                    if model_name == "ARIMA":
                        model, performance = self._train_arima(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "SARIMA":
                        return self._train_sarima(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "LinearRegression":
                        return self._train_linear_regression(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "SVR":
                        return self._train_svr(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "GradientBoosting":
                        return self._train_gradient_boosting(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "ExponentialSmoothing":
                        model, performance = self._train_exponential_smoothing(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "RandomForest":
                        model, performance = self._train_random_forest(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "LSTM" and self.config.enable_lstm:
                        model, performance = self._train_lstm(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "Prophet" and self.config.enable_prophet:
                        model, performance = self._train_prophet(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "BMTGNN" and BMTGNN_AVAILABLE:
                        model, performance = self._train_bmtgnn(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "VisionTransformer":
                        model, performance = self._train_vision_transformer(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "SimpleVisionTransformer":
                        model, performance = self._train_simple_vision_transformer(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "SpatioTemporalEnsemble" and ENSEMBLE_AVAILABLE:
                        model, performance = self._train_spatiotemporal_ensemble(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "VisionTransformerForTimeSeries" and ENSEMBLE_AVAILABLE:
                        model, performance = self._train_vision_transformer_timeseries(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "SimpleGraphTransformer":
                        model, performance = self._train_simple_graph_transformer(train_data, test_data, model_config["hyperparameters"])
                    elif model_name == "GraphTransformer":
                        model, performance = self._train_graph_transformer(train_data, test_data, model_config["hyperparameters"])
                    else:
                        continue

                    node_models[model_name] = model
                    node_performance[model_name] = performance
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {node}: {e}")
            
            trained_models[node] = node_models
            model_performance[node] = node_performance
        
        return {
            "trained_models": trained_models,
            "performance": model_performance
        }
    
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
            
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return model, metrics
            
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
            
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return model, metrics
            
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

    def _train_lstm(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train LSTM model"""
        try:
            sequence_length = hyperparams.get("sequence_length", 12)
            hidden_dim = hyperparams.get("hidden_dim", 50)
            num_layers = hyperparams.get("num_layers", 2)
            epochs = hyperparams.get("epochs", 100)
            
            lstm_model = LSTMForecaster(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1
            )
            
            lstm_model.train(train_data, epochs=epochs, sequence_length=sequence_length)
            
            # Evaluate on test data
            horizon = len(test_data)
            forecast = lstm_model.forecast(train_data, horizon, sequence_length)
            
            # Ensure forecast length matches test data
            if len(forecast) > len(test_data):
                forecast = forecast[:len(test_data)]
            elif len(forecast) < len(test_data):
                forecast = np.pad(forecast, (0, len(test_data) - len(forecast)), mode='edge')
            
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return lstm_model, metrics
            
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return None, {}
        
    def _train_arima(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train ARIMA model"""
        best_model = None
        best_aic = np.inf
        
        # Grid search over hyperparameters
        for p in hyperparams.get("p", [0,1]):
            for d in hyperparams.get("d", [0,1]):
                for q in hyperparams.get("q", [0,1]):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        # Evaluate on test data
        if best_model:
            forecast = best_model.forecast(steps=len(test_data))
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return best_model, metrics
        
        return None, {}
    
    def _train_exponential_smoothing(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Exponential Smoothing model"""
        try:
            model = ExponentialSmoothing(
                train_data,
                trend=hyperparams.get("trend", "add"),
                seasonal=hyperparams.get("seasonal", "add"),
                seasonal_periods=self.config.seasonal_period
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(len(test_data))
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return fitted_model, metrics
        except Exception as e:
            logger.error(f"ExponentialSmoothing error: {e}")
            return None, {}
        
    def _train_random_forest(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Random Forest model for time series"""
        try:
            # Create features for time series
            X_train, y_train = self._create_features(train_data)
            X_test, y_test = self._create_features(test_data)
            
            model = RandomForestRegressor(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", 10),
                random_state=42
            )
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            metrics = self._calculate_enhanced_metrics(y_test, predictions)
            
            return model, metrics
        except Exception as e:
            logger.error(f"RandomForest error: {e}")
            return None, {}
        
    def _train_prophet(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            train_df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            test_df = pd.DataFrame({
                'ds': test_data.index,
                'y': test_data.values
            })
            
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=hyperparams.get("yearly_seasonality", True),
                weekly_seasonality=hyperparams.get("weekly_seasonality", False),
                daily_seasonality=hyperparams.get("daily_seasonality", False),
                changepoint_prior_scale=hyperparams.get("changepoint_prior_scale", 0.05)
            )
            
            model.fit(train_df)
            
            # Create future dataframe for test period
            future = model.make_future_dataframe(periods=len(test_data), freq='M')
            forecast_df = model.predict(future)
            
            # Extract predictions for test period
            test_forecast = forecast_df.tail(len(test_data))['yhat'].values
            
            metrics = self._calculate_enhanced_metrics(test_data.values, test_forecast)
            return model, metrics
            
        except Exception as e:
            logger.error(f"Prophet training error: {e}")
            return None, {}

    def _train_vision_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Vision Transformer model using the architecture from """
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
            
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return model, metrics
            
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
            
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            return model, metrics
            
        except Exception as e:
            logger.error(f"SimpleVisionTransformer training error: {e}")
            # Return fallback
            return None, {}
    
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
        
    def _train_bmtgnn(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train Bayesian MTGNN model using the exact architecture"""
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
                # Use the test window approach similar to original script
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
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model, metrics
            
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
        """Train SimpleGraphTransformer model """
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
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"SimpleGraphTransformer training error: {e}")
            # Clean up temporary file if it exists
            import os
            temp_data_path = "temp_simple_gt_data.csv"
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            return None, {}

    def _train_graph_transformer(self, train_data: pd.Series, test_data: pd.Series, hyperparams: Dict) -> Tuple[Any, Dict]:
        """Train GraphTransformer model """
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
            metrics = self._calculate_enhanced_metrics(test_data.values, forecast)
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"GraphTransformer training error: {e}")
            # Clean up temporary file if it exists
            import os
            temp_data_path = "temp_gt_data.csv"
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
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
            
    def _create_features(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for machine learning models"""
        lags = 3
        X, y = [], []
        
        for i in range(lags, len(series)):
            X.append(series.iloc[i-lags:i].values)
            y.append(series.iloc[i])
        
        return np.array(X), np.array(y)    
     
    
    def _calculate_enhanced_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced metrics including cybersecurity-specific metrics"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return {}
        
        # Standard metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        smape = 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        
        # Relative errors
        rae = np.sum(np.abs(predicted - actual)) / np.sum(np.abs(actual - np.mean(actual)))
        rrse = np.sqrt(np.sum((predicted - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        # Cybersecurity-specific metrics
        threshold = np.percentile(actual, 70)  # Use 70th percentile as threat threshold
        
        false_positive_rate = self.calculate_false_positive_rate(predicted, actual, threshold)
        attack_coverage = self.calculate_attack_coverage(predicted, actual, threshold)
        alert_precision = self.calculate_alert_precision(predicted, actual, threshold)
        directional_accuracy = self.calculate_directional_accuracy(predicted, actual)
        temporal_correlation = self.calculate_temporal_correlation(predicted, actual)
        adversarial_robustness = self.calculate_adversarial_robustness(predicted, actual)
        
        # Quantile loss
        quantile_losses = self.calculate_quantile_loss(predicted, actual)
        
        # Operational Readiness Index
        recall = attack_coverage  # Using coverage as recall proxy
        resource_efficiency = 1 - false_positive_rate  # Lower FPR = better resource usage
        operational_readiness = self.calculate_operational_readiness_index(
            alert_precision, recall, false_positive_rate, resource_efficiency
        )
        
        return {
                    # Standard metrics
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "RRSE": rrse,
                    "RAE": rae,
                    "MAPE": mape,
                    "S_MAPE": smape,
                    
                    # Cybersecurity metrics
                    "False_Positive_Rate": false_positive_rate,
                    "Attack_Coverage": attack_coverage,
                    "Alert_Precision": alert_precision,
                    "Directional_Accuracy": directional_accuracy,
                    "Temporal_Correlation": temporal_correlation,
                    "Adversarial_Robustness": adversarial_robustness,
                    "Operational_Readiness_Index": operational_readiness,
                    
                    # Quantile losses
                    "Quantile_Losses": quantile_losses 
        }
    
    # Custom evaluation metrics implementation
    def calculate_false_positive_rate(self, predictions, actuals, threshold=0.3):
        """Calculate False Positive Rate for threat detection"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        fp = np.sum((pred_binary == 1) & (actual_binary == 0))
        tn = np.sum((pred_binary == 0) & (actual_binary == 0))
        
        fpr = fp / (fp + tn + 1e-8)
        return fpr
    
    def calculate_attack_coverage(self, predictions, actuals, threshold=0.3):
        """Calculate Attack Coverage percentage"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        total_attacks = np.sum(actuals > threshold)
        detected_attacks = np.sum((predictions > threshold) & (actuals > threshold))
        
        coverage = detected_attacks / (total_attacks + 1e-8)
        return coverage
    
    def calculate_alert_precision(self, predictions, actuals, threshold=0.3):
        """Calculate Alert Precision"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        true_positives = np.sum((predictions > threshold) & (actuals > threshold))
        false_positives = np.sum((predictions > threshold) & (actuals <= threshold))
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        return precision
    
    def calculate_directional_accuracy(self, predictions, actuals):
        """Calculate Directional Accuracy for trend prediction"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        # For single-step predictions
        if len(predictions.shape) == 1:
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
        else:
            # For multi-step predictions
            pred_direction = np.diff(predictions, axis=0) > 0
            actual_direction = np.diff(actuals, axis=0) > 0
        
        if pred_direction.size == 0:
            return 0.0
            
        correct_direction = (pred_direction == actual_direction)
        directional_accuracy = np.mean(correct_direction)
        return directional_accuracy
    
    def calculate_temporal_correlation(self, predictions, actuals):
        """Calculate Temporal Correlation Coefficient"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        # Flatten arrays for correlation calculation
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        if np.std(actual_flat) > 0 and np.std(pred_flat) > 0:
            corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
            if not np.isnan(corr):
                return corr
        
        return 0.0
    
    def calculate_adversarial_robustness(self, predictions, actuals, noise_level=0.1):
        """Calculate Adversarial Robustness Score"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        # Add small perturbations to predictions
        noisy_predictions = predictions + np.random.normal(0, noise_level, predictions.shape)
        
        # Calculate change in predictions relative to actuals
        original_error = np.abs(predictions - actuals)
        noisy_error = np.abs(noisy_predictions - actuals)
        
        error_increase = np.mean(noisy_error - original_error)
        robustness_score = 1.0 / (1.0 + error_increase)
        
        return max(0, min(1, robustness_score))
    
    def calculate_quantile_loss(self, predictions, actuals, quantiles=[0.1, 0.5, 0.9]):
        """Calculate Quantile Loss (Pinball Loss)"""
        if predictions.size == 0 or actuals.size == 0:
            return {f'quantile_{int(q*100)}': float('inf') for q in quantiles}
            
        losses = {}
        
        for q in quantiles:
            error = actuals - predictions
            loss = np.maximum(q * error, (q - 1) * error)
            losses[f'quantile_{int(q*100)}'] = np.mean(loss)
        
        return losses
    
    def calculate_operational_readiness_index(self, precision, recall, fpr, resource_efficiency):
        """Calculate composite Operational Readiness Index"""
        # Normalize metrics to 0-1 scale
        precision_norm = precision
        recall_norm = recall
        fpr_norm = 1 - fpr  # Lower FPR is better
        resource_norm = min(resource_efficiency, 1.0)
        
        # Weighted combination focusing on operational effectiveness
        ori = (0.3 * precision_norm + 0.3 * recall_norm + 
               0.2 * fpr_norm + 0.2 * resource_norm)
        
        return ori

class EnhancedCyberSecurityForecaster:
    """Enhanced cybersecurity forecaster with XAI, RAG, advanced models, and PAT recommendations"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.forecasting_engine = EnhancedForecastingEngine(config)
        self.visualization_agent = VisualizationAgent(config)
        self.xai_analyzer = XAIAnalyzer(config)
        self.report_agent = ReportAgent(config)

        # Initialize Preprocess Agent if enabled
        if config.enable_preprocessing:
            self.preprocess_agent = PreprocessAgent(
                model=config.llm_model,
                config=config.preprocess_config
            )
        else:
            self.preprocess_agent = None
        
        # Initialize PAT Recommendation Agent
        self.pat_config = PATConfig(
            graph_path=config.graph_path,
            recommendation_threshold=0.6,
            max_recommendations=5,
            cost_optimization=True,
            effectiveness_weight=0.6,
            cost_weight=0.3,
            implementation_weight=0.1,
            enable_llm_optimization=True
        )
        self.pat_agent = PATRecommendationAgent(self.pat_config, self.forecasting_engine.llm_manager)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.xai_dir = self.output_dir / "xai_analysis"
        self.xai_dir.mkdir(exist_ok=True)
        self.pat_dir = self.output_dir / "pat_recommendations"
        self.pat_dir.mkdir(exist_ok=True)
        self.preprocess_dir = self.output_dir / "preprocessing"
        self.preprocess_dir.mkdir(exist_ok=True)
    
    def run_enhanced_forecast_pipeline(self) -> Dict[str, Any]:
        """Run enhanced forecasting pipeline with XAI, RAG, and PAT recommendations"""
        logger.info("Starting enhanced cybersecurity forecasting pipeline...")
        
        # Step 1: Data loading and preprocessing (with RAG augmentation)
        data, node_names = self.forecasting_engine.data_agent.load_and_validate_data(
            self.config.data_path, self.config.graph_path
        )
        
        # Step 1A: Advanced preprocessing if enabled
        if self.config.enable_preprocessing and self.preprocess_agent:
            logger.info("Running advanced preprocessing with PreprocessAgent...")
            preprocess_results = self._run_advanced_preprocessing(data, node_names)
            processed_data = preprocess_results["cleaned_data"]
            preprocessing_report = preprocess_results["analysis_report"]
            preprocessing_visualizations = preprocess_results["visualizations"]
        else:
            # Use existing preprocessing
            processed_data_dict = self.forecasting_engine.data_agent.preprocess_data(data, node_names)
            processed_data = processed_data_dict["data"]
            preprocessing_report = {}
            preprocessing_visualizations = {}
            
        # Augment analysis with threat intelligence (only if RAG enabled)
        analysis_context = f"Cybersecurity time series data with {len(node_names)} nodes, shape {data.shape}"
            
        # Step 2: Time series analysis 
        processed_data = self.forecasting_engine.data_agent.preprocess_data(data, node_names)

        ts_data = self.forecasting_engine.prepare_time_series_data(
            processed_data["data"], processed_data["node_names"]
        )
        
        analysis_results = self.forecasting_engine.analysis_agent.analyze_time_series(
            processed_data["data"], processed_data["node_names"]
        )
        
        # Step 3: Enhanced model selection with comprehensive validation
        logger.info("Running comprehensive model validation...")
        validation_results = self.forecasting_engine.run_comprehensive_validation(
            analysis_results, processed_data
        )
        # Step 4: Model training with enhanced metrics
        best_models = validation_results.get("best_models", [])
        model_selection = {"selected_models": best_models}
        training_results = self.forecasting_engine.train_models(ts_data, model_selection)
        
        # Step 5: Generate forecasts
        forecasts = self._generate_final_forecasts(training_results, ts_data)
        
        # Step 6: XAI Analysis if enabled
        if self.config.enable_xai:
            xai_results = self._perform_xai_analysis(training_results, processed_data["data"], node_names)
        else:
            xai_results = {}
        
        # Step 7: PAT Recommendations based on forecasts
        #pat_recommendations = self._generate_pat_recommendations(forecasts, node_names)
        
        # Step 8: Generate comprehensive report using Report Agent
        logger.info("Generating comprehensive final report...")
        comprehensive_report = self.report_agent.generate_comprehensive_report({
            "forecasts": forecasts,
            "training_results": training_results,
            "analysis_results": analysis_results,
            "xai_results": xai_results,
            #"pat_recommendations": pat_recommendations,
            "preprocessing_results": preprocessing_report,
            "model_selection": model_selection,
            "validation_results": validation_results
        })
        
        # Step 9: Generate enhanced outputs including comprehensive report
        self._generate_enhanced_outputs(forecasts, training_results, analysis_results, 
                                      xai_results, 
                                      #pat_recommendations, 
                                      node_names,
                                      preprocessing_report,
                                      comprehensive_report)
        
        logger.info("Enhanced forecasting pipeline completed successfully!")
        
        return {"forecasts": forecasts,
            "training_results": training_results,
            "analysis_results": analysis_results,
            "xai_results": xai_results,
            #"pat_recommendations": pat_recommendations,
            "preprocessing_results": preprocessing_report,
            "model_selection": model_selection,
            "comprehensive_report": comprehensive_report}

    def _run_advanced_preprocessing(self, data: pd.DataFrame, node_names: List[str]) -> Dict[str, Any]:
        """Run advanced preprocessing using PreprocessAgent"""
        logger.info("Running advanced preprocessing pipeline...")
        
        # Prepare data for preprocessing (ensure it has the expected format)
        preprocess_data = self._prepare_data_for_preprocessing(data, node_names)
        
        # Run preprocessing agent
        preprocess_results = self.preprocess_agent.process(
            preprocess_data, 
            str(self.preprocess_dir)
        )
        
        # Convert back to original format if needed
        cleaned_data = self._convert_preprocessed_data(
            preprocess_results["cleaned_data"], 
            data, 
            node_names
        )
        
        preprocess_results["cleaned_data"] = cleaned_data
        
        # Save preprocessing summary
        self._save_preprocessing_summary(preprocess_results)
        
        return preprocess_results
    
    
    def _prepare_data_for_preprocessing(self, data: pd.DataFrame, node_names: List[str]) -> pd.DataFrame:
        """Prepare data for preprocessing agent (convert to expected format)"""
        # The preprocessing agent expects a DataFrame with 'value' column
        # If our data has multiple columns, we might need to process each column separately
        
        if len(data.columns) == 1:
            # Single column - rename to 'value'
            preprocess_data = data.copy()
            preprocess_data.columns = ['value']
        else:
            # Multiple columns - use first column or create composite
            # For cybersecurity data, we might want to process each threat type separately
            # Here we'll use the first column as an example
            preprocess_data = pd.DataFrame({
                'value': data.iloc[:, 0]  # Use first column
            })
        
        # Ensure index is proper for time series
        if not isinstance(preprocess_data.index, pd.DatetimeIndex):
            # Create a time index if not present
            preprocess_data.index = pd.date_range(
                start='2020-01-01', 
                periods=len(preprocess_data), 
                freq='M'
            )
        
        return preprocess_data
    
    def _convert_preprocessed_data(self, preprocessed_data: pd.DataFrame, 
                                 original_data: pd.DataFrame, 
                                 node_names: List[str]) -> pd.DataFrame:
        """Convert preprocessed data back to original format"""
        # This method converts the preprocessed data (with 'value' column) back to the original multi-column format
        
        cleaned_data = original_data.copy()
        
        # Apply the same preprocessing to all columns
        for i, col in enumerate(cleaned_data.columns):
            if i == 0:  # For the first column, use the preprocessed values
                cleaned_data[col] = preprocessed_data['value'].values
            else:
                # For other columns, apply similar preprocessing
                cleaned_data[col] = cleaned_data[col].ffill().bfill()  # Simple fill
        
        return cleaned_data
    
    def _save_preprocessing_summary(self, preprocess_results: Dict[str, Any]):
        """Save preprocessing summary to file"""
        summary_path = self.preprocess_dir / "preprocessing_summary.json"
        
        summary = {
            "preprocessing_report": preprocess_results.get("analysis_report", {}),
            "preprocessing_config": preprocess_results.get("preprocess_config", {}),
            "validation_result": preprocess_results.get("validation_result", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Preprocessing summary saved to: {summary_path}")
        
    
    def _generate_pat_recommendations(self, forecasts: Dict[str, Any], node_names: List[str]) -> Dict[str, Any]:
        """Generate PAT recommendations based on forecasted threats"""
        logger.info("Generating PAT recommendations for forecasted threats...")
        
        # Extract forecasted threats with probabilities
        forecasted_threats = self._extract_forecasted_threats(forecasts, node_names)
        
        if not forecasted_threats:
            logger.warning("No forecasted threats found for PAT recommendations")
            return {}
        
        # Generate PAT recommendations
        pat_recommendations = self.pat_agent.recommend_pats_for_threats(
            forecasted_threats,
            budget_constraint=100000,  # Example budget constraint
            time_constraint=6  # Example time constraint
        )
        
        # Generate PAT visualizations
        pat_viz_path = self.pat_dir / "pat_recommendations_analysis.png"
        self.pat_agent.visualize_recommendations(pat_recommendations, str(pat_viz_path))
        
        # Save PAT recommendations to file
        self._save_pat_recommendations(pat_recommendations)
        
        return pat_recommendations
    
    def _extract_forecasted_threats(self, forecasts: Dict[str, Any], node_names: List[str]) -> Dict[str, float]:
        """Extract forecasted threats and convert to probability scores"""
        forecasted_threats = {}
        
        for node, model_forecasts in forecasts.items():
            if model_forecasts:
                # Use the first available model's forecast for threat probability
                for model_name, forecast_values in model_forecasts.items():
                    if isinstance(forecast_values, list) and len(forecast_values) > 0:
                        # Convert forecast values to probability scores (0-1)
                        probability = self._convert_forecast_to_probability(forecast_values)
                        forecasted_threats[node] = probability
                        break  # Use first model only for simplicity
        
        return forecasted_threats
    
    def _convert_forecast_to_probability(self, forecast_values: List[float]) -> float:
        """Convert forecast values to probability scores (0-1)"""
        if not isinstance(forecast_values, list):
            return 0.0
        
        # Normalize forecast values to 0-1 range
        # This assumes forecast values represent threat intensity/severity
        abs_forecast = [abs(x) for x in forecast_values]
        max_val = max(abs_forecast) if max(abs_forecast) > 0 else 1.0
        normalized_forecast = [x / max_val for x in abs_forecast]
        
        # Use average of normalized values as probability
        probability = min(1.0, np.mean(normalized_forecast))
        
        return probability
    
    def _save_pat_recommendations(self, pat_recommendations: Dict[str, Any]):
        """Save PAT recommendations to JSON file"""
        pat_output_path = self.pat_dir / "pat_recommendations.json"
        
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        pat_recommendations_serializable = convert_types(pat_recommendations)
        
        with open(pat_output_path, 'w') as f:
            json.dump(pat_recommendations_serializable, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"PAT recommendations saved to: {pat_output_path}")
    
    def _perform_xai_analysis(self, training_results: Dict[str, Any], 
                            data: pd.DataFrame, node_names: List[str]) -> Dict[str, Any]:
        """Perform comprehensive XAI analysis"""
        logger.info("Performing comprehensive XAI analysis...")
        
        # Get trained models for the first node
        first_node = list(training_results["trained_models"].keys())[0]
        models = training_results["trained_models"][first_node]
        
        # Run comprehensive XAI analysis
        xai_results = self.xai_analyzer.comprehensive_xai_analysis(
            models, data, node_names
        )
        
        return xai_results
    
    def _generate_final_forecasts(self, training_results: Dict[str, Any], ts_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Generate final forecasts for all nodes including new models"""
        forecasts = {}
        
        for node, models in training_results["trained_models"].items():
            node_forecasts = {}
            
            for model_name, model in models.items():
                try:
                    if model_name == "ARIMA":
                        forecast = model.forecast(self.config.forecast_months)
                    elif model_name == "ExponentialSmoothing":
                        forecast = model.forecast(self.config.forecast_months)
                    elif model_name == "RandomForest":
                        # For RF, use last values for prediction
                        last_values = ts_data[node].iloc[-3:].values
                        forecast = [model.predict([last_values])[0]] * self.config.forecast_months
                    elif model_name == "LSTM":
                        forecast = model.forecast(ts_data[node], self.config.forecast_months)
                    elif model_name == "Prophet":
                        # Create future dataframe for Prophet
                        future_dates = pd.date_range(
                            start=ts_data[node].index[-1] + pd.DateOffset(months=1),
                            periods=self.config.forecast_months,
                            freq='M'
                        )
                        future_df = pd.DataFrame({'ds': future_dates})
                        prophet_forecast = model.predict(future_df)
                        forecast = prophet_forecast['yhat'].values
                    elif model_name == "BMTGNN":
                        forecast = self._predict_bmtgnn(model, ts_data[node], {})
                    elif model_name == "VisionTransformer":
                        # Vision Transformer forecast
                        input_seq = torch.FloatTensor(ts_data[node].values[-24:]).unsqueeze(0).unsqueeze(-1)
                        model.eval()
                        with torch.no_grad():
                            forecast = model(input_seq).squeeze().numpy()[:self.config.forecast_months]
                    elif model_name == "SimpleVisionTransformer":
                        # Simple Vision Transformer forecast  
                        input_seq = torch.FloatTensor(ts_data[node].values[-12:]).unsqueeze(0).unsqueeze(-1)
                        model.eval()
                        with torch.no_grad():
                            forecast = model(input_seq).squeeze().numpy()[:self.config.forecast_months]
                    elif model_name == "SpatioTemporalEnsemble" and ENSEMBLE_AVAILABLE:
                        # SpatioTemporalEnsemble forecast
                        input_seq = torch.FloatTensor(ts_data[node].values[-model.config['sequence_length']:]).unsqueeze(0)
                        model.eval()
                        with torch.no_grad():
                            forecast = model(input_seq).squeeze().numpy()[:self.config.forecast_months]
                    elif model_name == "VisionTransformerForTimeSeries" and ENSEMBLE_AVAILABLE:
                        # VisionTransformerForTimeSeries forecast
                        input_seq = torch.FloatTensor(ts_data[node].values[-model.config['sequence_length']:]).unsqueeze(0)
                        model.eval()
                        with torch.no_grad():
                            forecast = model(input_seq).squeeze().numpy()[:self.config.forecast_months]
                    elif model_name == "SimpleGraphTransformer":
                    # SimpleGraphTransformer forecast
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        with torch.no_grad():
                            # Create input data for the model
                            input_seq = torch.FloatTensor(ts_data[node].values[-model.config['sequence_length']:]).unsqueeze(0).unsqueeze(-1)
                            forecast = np.full(self.config.forecast_months, ts_data[node].iloc[-1])
                    elif model_name == "GraphTransformer":
                        # GraphTransformer forecast
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        with torch.no_grad():
                            # Create input data for the model
                            input_seq = torch.FloatTensor(ts_data[node].values[-model.config['sequence_length']:]).unsqueeze(0).unsqueeze(-1)
                            forecast = np.full(self.config.forecast_months, ts_data[node].iloc[-1])
                    else:
                        continue
                    
                    node_forecasts[model_name] = forecast
                except Exception as e:
                    logger.error(f"Forecast generation failed for {model_name} on {node}: {e}")
            
            forecasts[node] = node_forecasts
        
        return forecasts
    
    def _generate_enhanced_outputs(self, forecasts: Dict[str, Any], training_results: Dict[str, Any],
                                 analysis_results: Dict[str, Any], xai_results: Dict[str, Any],
                                 pat_recommendations: Dict[str, Any], node_names: List[str],
                                 preprocessing_report: Dict[str, Any] = None,
                                 preprocessing_visualizations: Dict[str, str] = None):
        """Generate all enhanced output files and visualizations including PAT recommendations"""
        
        # Generate standard plots
        for node in list(forecasts.keys())[:645]:
            if node in training_results["performance"]:
                # Forecast comparison plot
                plot_path = self.output_dir / f"forecast_comparison_{node.replace(' ', '_')}.png"
                self.visualization_agent.plot_forecast_comparison(
                    self.forecasting_engine.prepare_time_series_data(pd.DataFrame(), [node])[node],
                    forecasts[node],
                    node,
                    str(plot_path)
                )
                
                # Metrics comparison plot
                metrics_path = self.output_dir / f"metrics_comparison_{node.replace(' ', '_')}.png"
                self.visualization_agent.plot_metrics_comparison(
                    training_results["performance"][node],
                    node,
                    str(metrics_path)
                )
        
        # Generate PAT-specific outputs if available
        if pat_recommendations:
            self._generate_pat_outputs(pat_recommendations, forecasts)
        
        # Save enhanced results to JSON
        results = {
            "forecasts": forecasts,
            "performance_metrics": training_results["performance"],
            "analysis_insights": analysis_results.get("llm_insights", {}),
            "xai_results": xai_results,
            "pat_recommendations": pat_recommendations,
            "preprocessing_results": preprocessing_report,
            "model_selection": training_results.get("model_selection", {}),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "forecast_months": self.config.forecast_months,
                "llm_model": self.config.llm_model,
                "llm_provider": self.config.llm_provider,
                "enable_xai": self.config.enable_xai,
                "enable_rag": self.config.enable_rag,
                "enable_preprocessing": self.config.enable_preprocessing,
                "pat_recommendations": True
            }
        }
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results = convert_types(results)
        
        with open(self.output_dir / "enhanced_forecast_results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Generate enhanced summary report including PAT recommendations
        self._generate_enhanced_summary_report(results, node_names, xai_results, pat_recommendations)
    
    def _generate_pat_outputs(self, pat_recommendations: Dict[str, Any], forecasts: Dict[str, Any]):
        """Generate PAT-specific output files and reports"""
        
        # Generate PAT implementation summary
        pat_summary_path = self.pat_dir / "pat_implementation_summary.txt"
        self._generate_pat_implementation_summary(pat_recommendations, pat_summary_path)
        
        # Generate PAT-threat mapping
        pat_mapping_path = self.pat_dir / "pat_threat_mapping.csv"
        self._generate_pat_threat_mapping(pat_recommendations, forecasts, pat_mapping_path)
        
        logger.info(f"PAT-specific outputs saved to: {self.pat_dir}")
    
    def _generate_pat_implementation_summary(self, pat_recommendations: Dict[str, Any], output_path: Path):
        """Generate detailed PAT implementation summary"""
        if not pat_recommendations.get('recommended_pats'):
            return
        
        best_combo = pat_recommendations['recommended_pats'][0]
        risk_metrics = pat_recommendations.get('risk_reduction_metrics', {})
        coverage_analysis = pat_recommendations.get('coverage_analysis', {})
        
        summary = [
            "PAT IMPLEMENTATION SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RECOMMENDED PAT COMBINATION:",
            "-" * 30,
            f"Combination ID: {best_combo.get('combination_id', 'N/A')}",
            f"PATs: {', '.join(best_combo.get('pats', []))}",
            f"Composite Score: {best_combo.get('composite_score', 0):.3f}",
            f"Total Cost: ${best_combo.get('total_cost', 0):,.2f}",
            f"Implementation Time: {best_combo.get('implementation_time', 0)} months",
            "",
            "RISK REDUCTION ANALYSIS:",
            "-" * 30,
            f"Initial Risk Score: {risk_metrics.get('initial_risk_score', 0):.2f}",
            f"Residual Risk Score: {risk_metrics.get('residual_risk_score', 0):.2f}",
            f"Risk Reduction: {risk_metrics.get('risk_reduction_percentage', 0):.1f}%",
            f"Risk Reduction Efficiency: {risk_metrics.get('risk_reduction_efficiency', 0):.2f}",
            "",
            "COVERAGE ANALYSIS:",
            "-" * 30,
            f"Total Threats: {coverage_analysis.get('total_threats', 0)}",
            f"Covered Threats: {coverage_analysis.get('covered_threats', 0)}",
            f"Coverage Percentage: {coverage_analysis.get('coverage_percentage', 0):.1f}%",
            f"High Risk Coverage: {coverage_analysis.get('high_risk_coverage', 0):.1f}%",
            "",
            "IMPLEMENTATION ROADMAP:",
            "-" * 30
        ]
        
        roadmap = pat_recommendations.get('implementation_roadmap', [])
        for i, step in enumerate(roadmap, 1):
            summary.extend([
                f"{i}. {step.get('pat', 'N/A')}",
                f"   Phase: {step.get('phase', 'N/A')}",
                f"   Timeline: {step.get('estimated_timeline', 'N/A')}",
                f"   Resources: Team of {step.get('required_resources', {}).get('team_size', 0)}",
                ""
            ])
        
        # Add critical gaps if any
        critical_gaps = coverage_analysis.get('critical_gaps', [])
        if critical_gaps:
            summary.extend([
                "CRITICAL COVERAGE GAPS:",
                "-" * 30
            ])
            for gap in critical_gaps[:5]:  # Show top 5 gaps
                summary.append(f"- {gap}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(summary))
    
    def _generate_pat_threat_mapping(self, pat_recommendations: Dict[str, Any], 
                                   forecasts: Dict[str, Any], output_path: Path):
        """Generate CSV mapping of PATs to covered threats"""
        if not pat_recommendations.get('recommended_pats'):
            return
        
        best_combo = pat_recommendations['recommended_pats'][0]
        covered_threats = best_combo.get('covered_threats', [])
        
        # Create mapping data
        mapping_data = []
        for threat_data in covered_threats:
            threat_name = threat_data.get('threat', '')
            probability = threat_data.get('probability', 0)
            effectiveness = threat_data.get('effectiveness', 0)
            
            # Find which PATs cover this threat
            covering_pats = []
            for pat in best_combo.get('pats', []):
                # Check if this PAT covers the threat
                if threat_name in [t.get('threat', '') for t in pat_recommendations.get('threat_analysis', {}).keys()]:
                    covering_pats.append(pat)
            
            mapping_data.append({
                'Threat': threat_name,
                'Probability': probability,
                'Effectiveness': effectiveness,
                'Covering_PATs': ', '.join(covering_pats),
                'Risk_Score': probability * (1 - effectiveness)  # Residual risk
            })
        
        # Create DataFrame and save to CSV
        if mapping_data:
            df = pd.DataFrame(mapping_data)
            df.to_csv(output_path, index=False)
    
    def _generate_enhanced_summary_report(self, results: Dict[str, Any], node_names: List[str], 
                                        xai_results: Dict[str, Any], pat_recommendations: Dict[str, Any], preprocessing_report: Dict[str, Any] = None):
        """Generate enhanced summary report with XAI insights and PAT recommendations"""
        report = [
            "ENHANCED CYBERSECURITY TIME SERIES FORECASTING REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Forecast Horizon: {self.config.forecast_months} months",
            f"LLM Model: {self.config.llm_model} ({self.config.llm_provider})",
            f"Nodes Analyzed: {len(node_names)}",
            f"XAI Enabled: {self.config.enable_xai}",
            f"RAG Enabled: {self.config.enable_rag}",
            f"Advanced Preprocessing: {self.config.enable_preprocessing}",
            f"PAT Recommendations: {'Generated' if pat_recommendations else 'None'}",
            "",
            "PERFORMANCE SUMMARY:",
            "-" * 30
        ]
        
        # Add preprocessing summary if available
        if preprocessing_report:
            report.extend([
                "",
                "PREPROCESSING SUMMARY:",
                "-" * 30
            ])
            
            quality_assessment = preprocessing_report.get('quality_assessment', {})
            data_overview = preprocessing_report.get('data_overview', {})
            
            if quality_assessment:
                report.append(f"Data Quality Score: {quality_assessment.get('data_quality_score', 'N/A')}")
            
            if data_overview and 'basic_stats' in data_overview:
                stats = data_overview['basic_stats']
                report.append(f"Data Statistics - Mean: {stats.get('mean', 'N/A'):.4f}, "
                            f"Std: {stats.get('std', 'N/A'):.4f}")
            
            for node, metrics in results["performance_metrics"].items():
                report.append(f"\n{node}:")
                for model, model_metrics in metrics.items():
                    report.append(f"  {model}:")
                    # Standard metrics
                    for metric in ["MAE", "RMSE", "MAPE", "RAE", "RRSE"]:
                        if metric in model_metrics:
                            report.append(f"    {metric}: {model_metrics[metric]:.4f}")
                    # Cybersecurity metrics
                    for metric in ["False_Positive_Rate", "Attack_Coverage", "Alert_Precision", "Operational_Readiness_Index"]:
                        if metric in model_metrics:
                            report.append(f"    {metric}: {model_metrics[metric]:.4f}")
        
        # Add PAT Recommendations section
        if pat_recommendations and pat_recommendations.get('recommended_pats'):
            best_combo = pat_recommendations['recommended_pats'][0]
            risk_metrics = pat_recommendations.get('risk_reduction_metrics', {})
            
            report.extend([
                "",
                "PAT RECOMMENDATIONS:",
                "-" * 30,
                f"Top PAT Combination: {', '.join(best_combo.get('pats', []))}",
                f"Risk Reduction: {risk_metrics.get('risk_reduction_percentage', 0):.1f}%",
                f"Total Cost: ${best_combo.get('total_cost', 0):,.2f}",
                f"Implementation Time: {best_combo.get('implementation_time', 0)} months",
                ""
            ])
        
        # Add XAI insights
        if xai_results:
            report.extend([
                "",
                "EXPLAINABLE AI INSIGHTS:",
                "-" * 30
            ])
            
            if "multi_model_consensus" in xai_results:
                consensus = xai_results["multi_model_consensus"]
                report.append("Multi-Model Consensus Analysis:")
                report.append(f"  High Agreement Features: {', '.join(consensus.get('high_agreement_features', [])[:5])}")
                report.append(f"  Disagreement Features: {', '.join(consensus.get('disagreement_features', [])[:5])}")
            
            if "anchors_explanations" in xai_results:
                anchors = xai_results["anchors_explanations"]
                report.append("Rule-Based Explanations:")
                for i, rule in enumerate(anchors.get("decision_rules", [])[:3]):
                    report.append(f"  Rule {i+1}: {rule}")
        
        report.extend([
            "",
            "KEY INSIGHTS:",
            "-" * 30,
            results["analysis_insights"].get("overall_patterns", "No insights available"),
            "",
            "RECOMMENDED MODELS:",
            "-" * 30
        ])
        
        for model in results["analysis_insights"].get("recommended_models", []):
            report.append(f"  - {model}")
        
        with open(self.output_dir / "enhanced_summary_report.txt", "w") as f:
            f.write("\n".join(report))
            
def main():
    """Main execution function with enhanced capabilities"""
    
    # Configuration
    config = ForecastConfig(
        llm_provider = "ollama",
        llm_model = "deepseek-r1:8b",
        data_path="Dissertation/Agentic LLM/data/sm_data_g.csv",
        graph_path="Dissertation/Agentic LLM/data/graph.csv",
        forecast_months=36,
        output_dir="enhanced_cybersecurity_forecast_results",
        enable_xai=True,
        enable_rag=False,
        enable_lstm=True,
        enable_prophet=True,
        
        xai_methods=[  
            "shap", "lime", "attention", "permutation", "counterfactual",
            "causal", "faithfulness", "dynamic_weights", "dice", "consensus", "anchors"
        ],        
        preprocess_config={
            'preprocess': {
                'outlier_threshold': 1.5,
                'enable_llm_preprocessing': True
            }
        }
    )
    
    # Initialize and run enhanced forecaster
    forecaster = EnhancedCyberSecurityForecaster(config)
    
    try:
        results = forecaster.run_enhanced_forecast_pipeline()
        logger.info(f"Enhanced results saved to: {forecaster.output_dir}")
        
        xai_results = results.get('xai_results', {})
        successful_xai = len([r for r in xai_results.values() 
                            if not isinstance(r, dict) or r.get('status') not in ['error', 'unknown_method']])
        # Print enhanced summary
        print("\n" + "="*70)
        print("ENHANCED CYBERSECURITY FORECASTING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results directory: {forecaster.output_dir}")

        used_models = []
        for node_models in results['forecasts'].values():
            used_models.extend(node_models.keys())
        used_models = list(set(used_models))
        print(f"Models used: {used_models}")
        
        print(f"Forecasts generated for: {len(results['forecasts'])} nodes")
        print(f"Models used: {[m['model'] for m in results.get('model_selection', {}).get('selected_models', [])]}")
        print(f"XAI Analysis: {successful_xai}/{len(xai_results)} methods successful")
        print(f"XAI Analysis: {'Completed' if results.get('xai_results') else 'Disabled'}")
        print(f"RAG Augmentation: {'Enabled' if config.enable_rag else 'Disabled'}")
        print(f"Advanced Preprocessing: {'Completed' if results.get('preprocessing_results') else 'Disabled'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Enhanced forecasting pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()