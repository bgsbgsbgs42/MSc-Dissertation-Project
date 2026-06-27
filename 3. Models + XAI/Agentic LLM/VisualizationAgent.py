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

class VisualizationAgent:
    """Agent for generating plots and visualizations"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.set_plot_style()
    
    def set_plot_style(self):
        """Set matplotlib style"""
        plt.style.use(self.config.plot_style)
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
    
    def plot_forecast_comparison(self, actual: pd.Series, forecasts: Dict[str, np.ndarray], 
                               node_name: str, save_path: str):
        """Plot forecast comparison for a single node"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Full series with forecasts
        axes[0].plot(actual.index, actual.values, 'b-', label='Actual', linewidth=2)
        
        for model_name, forecast in forecasts.items():
            # Extend index for forecast period
            last_date = actual.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=len(forecast),
                freq='M'
            )
            axes[0].plot(forecast_dates, forecast, '--', label=f'{model_name} Forecast', alpha=0.8)
        
        axes[0].set_title(f'{node_name} - Forecast Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Zoom on recent data
        recent_data = actual.iloc[-24:]  # Last 2 years
        axes[1].plot(recent_data.index, recent_data.values, 'b-', label='Actual', linewidth=2)
        
        for model_name, forecast in forecasts.items():
            forecast_dates = pd.date_range(
                start=actual.index[-1] + pd.DateOffset(months=1),
                periods=len(forecast),
                freq='M'
            )
            axes[1].plot(forecast_dates, forecast, '--', label=f'{model_name} Forecast', alpha=0.8)
        
        axes[1].set_title('Recent Data & Forecast (Zoom)', fontsize=12)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]], node_name: str, save_path: str):
        """Plot metrics comparison across models"""
        models = list(metrics.keys())
        metric_names = ['MAE', 'RMSE', 'MAPE', 'RAE', 'RRSE']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_names):
            if i < len(axes):
                values = [metrics[model].get(metric, 0) for model in models]
                axes[i].bar(models, values, alpha=0.7)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for j, v in enumerate(values):
                    axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(len(metric_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Model Metrics Comparison - {node_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
