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
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import openai
from langchain_core.messages import HumanMessage, SystemMessage
#from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pgmpy.estimators import PC, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import scipy.stats as stats
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for the forecasting system"""
    # LLM Configuration
    llm_provider="ollama",
    llm_model="deepseek-r1:14b",  # or "mistral", "llama2", "codellama", etc.
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
        if self.config.llm_provider == "openai":
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
            )
        elif self.config.llm_provider == "ollama":
            return Ollama(
                model=self.config.llm_model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
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
        
        # Load graph metadata for node names - THESE BECOME TARGET COLUMNS
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
        3. Data normalization
        4. Feature engineering for cybersecurity context
        
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
        
        # Outlier detection (simplified)
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

class AnalysisAgent:
    """Agent for time series analysis and feature extraction"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    def analyze_time_series(self, data: pd.DataFrame, node_names: List[str]) -> Dict[str, Any]:
        """Comprehensive time series analysis"""
        logger.info("Performing time series analysis...")
        
        analysis_results = {}
        
        for i, node in enumerate(node_names[:5]):  # Analyze first 5 nodes for efficiency
            if i < data.shape[1]:
                ts_data = data.iloc[:, i]
                analysis_results[node] = self._analyze_single_series(ts_data, node)
        
        # LLM-guided overall analysis
        system_prompt = """You are a cybersecurity time series analysis expert. Provide insights about patterns, anomalies, and forecasting suitability."""
        
        user_prompt = f"""
        Analyse cybersecurity time series data with {len(node_names)} variables.
        Key statistics from first 5 series:
        { {node: analysis_results[node]['summary_stats'] for node in list(analysis_results.keys())[:3]} }
        
        Provide insights on:
        1. Overall data patterns
        2. Seasonality and trends
        3. Anomaly detection
        4. Forecasting challenges
        5. Recommended model types
        
        Return JSON format:
        {{
            "overall_patterns": "str",
            "seasonality_present": bool,
            "anomaly_likelihood": "str",
            "recommended_models": ["str"],
            "forecasting_confidence": "str"
        }}
        """
        
        llm_analysis = self.llm_manager.generate_response(system_prompt, user_prompt)
        analysis_results["llm_insights"] = self._parse_json_response(llm_analysis)
        
        return analysis_results
        
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

    def _analyze_single_series(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Analyze a single time series"""
        # Basic statistics
        stats = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "trend": self._calculate_trend(series),
            "stationarity": self._check_stationarity(series)
        }
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(series.dropna(), period=12, extrapolate_trend='freq')
            seasonal_strength = np.std(decomposition.seasonal) / np.std(series)
        except:
            seasonal_strength = 0
        
        return {
            "summary_stats": stats,
            "seasonal_strength": seasonal_strength,
            "is_stationary": stats["stationarity"]["p_value"] < 0.05
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Check stationarity using ADF test"""
        try:
            result = adfuller(series.dropna())
            return {
                "p_value": result[1],
                "stationary": result[1] < 0.05
            }
        except:
            return {"p_value": 1.0, "stationary": False}
