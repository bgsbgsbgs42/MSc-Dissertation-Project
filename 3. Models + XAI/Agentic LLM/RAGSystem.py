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
    enable_rag: bool = False
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
        system_prompt = """You are a cybersecurity data preprocessing expert. Analyze the data characteristics and recommend preprocessing strategies."""
        
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


class RAGSystem:
    """Retrieval-Augmented Generation system for cybersecurity threat intelligence"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._initialize_threat_database()
    
    def _initialize_threat_database(self):
        """Initialize or load threat intelligence database"""
        db_path = Path(self.config.rag_db_path)
        
        if db_path.exists() and (db_path / "index.faiss").exists():
            logger.info("Loading existing threat database...")
            self.vector_store = FAISS.load_local(
                str(db_path), self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            logger.info("Creating new threat database...")
            self._create_threat_database()
    
    def _create_threat_database(self):
        """
        Build a FAISS similarity index from unified_embeddings.csv so it integrates seamlessly with your existing RAG pipeline.
        
        """
        try:
            # Load unified embedding CSV
            unified_path = "data/unified_embeddings.csv"  # adjust if needed to a different file path
            df = pd.read_csv(unified_path)

            # Detect vector columns (all numeric columns)
            vector_cols = [c for c in df.columns if c not in ["text", "source_file"]]

            # Convert vectors to list-of-floats
            df["embedding"] = df[vector_cols].values.tolist()

    
            # Build LangChain Document objects
            documents = []
            for i, row in df.iterrows():
                documents.append(
                    Document(
                        page_content=str(row["text"]),
                        metadata={
                            "source": row.get("source_file", "unknown"),
                            "doc_id": i
                        }
                    )
                )

       
            # Build FAISS Vector Store
            print(f"Building FAISS index from {len(documents)} threat entries...")

            # NOTE: self.embeddings must be a LangChain embedding class 
            #       e.g. HuggingFaceEmbeddings, OpenAIEmbeddings, etc.
            # Build embedding arrays (ensure embeddings are numpy arrays)
            embedding_list = [np.array(e) for e in df["embedding"].tolist()]

            # Build text list for the required text_embeddings argument (robust to column name)
            if "text" in df.columns:
                text_list = df["text"].astype(str).tolist()
            elif "content" in df.columns:
                text_list = df["content"].astype(str).tolist()
            elif "description" in df.columns:
                text_list = df["description"].astype(str).tolist()
            else:
                # Fallback: use a string representation of the row or index
                text_list = df.index.astype(str).tolist()

            # Build metadatas (if present) else use row dicts
            if "metadata" in df.columns:
                metadatas = df["metadata"].tolist()
            else:
                metadatas = df.to_dict(orient="records")

            # Create FAISS vector store - provide both embeddings and text_embeddings
            # (FAISS.from_embeddings requires numeric embeddings + text_embeddings)
            vector_store = FAISS.from_embeddings(
                embeddings=embedding_list,
                text_embeddings=text_list,
                metadatas=metadatas,
                embedding=self.embeddings  # keep existing embedding object for search-time embedding
            )

            # Optionally store it as an attribute
            self.vector_store = vector_store

      
            # Save FAISS index locally
            db_path = Path(self.config.rag_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            vector_store.save_local(str(db_path))
            print(f"FAISS index saved to {db_path}")
            
        except FileNotFoundError as e:
            logger.warning(f"Unified embeddings file not found: {e}. RAG system will operate without threat database.")
            self.vector_store = None
        except Exception as e:
            logger.error(f"Error creating threat database: {e}. RAG system will operate without threat database.")
            self.vector_store = None

    
    def retrieve_relevant_threats(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant threat intelligence for a query"""
        if self.vector_store is None:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []
    
    def augment_analysis_with_rag(self, analysis_context: str) -> str:
        """Augment analysis with relevant threat intelligence"""
        threats = self.retrieve_relevant_threats(analysis_context)
        
        if not threats:
            return analysis_context
        
        rag_context = "\n\nRELEVANT THREAT INTELLIGENCE:\n"
        for i, threat in enumerate(threats, 1):
            rag_context += f"{i}. {threat}\n"
        
        return analysis_context + rag_context