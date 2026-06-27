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

class LSTMForecaster:
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 50, num_layers: int = 2, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()
    
    def _build_model(self) -> nn.Module:
        """Build LSTM model"""
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
                self.linear = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_time_step = lstm_out[:, -1, :]
                return self.linear(last_time_step)
        
        return LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
    
    def prepare_data(self, series: pd.Series, sequence_length: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for LSTM training"""
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        X, y = [], []
        for i in range(len(scaled_values) - sequence_length):
            X.append(scaled_values[i:(i + sequence_length)])
            y.append(scaled_values[i + sequence_length])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(self, series: pd.Series, epochs: int = 100, sequence_length: int = 12):
        """Train LSTM model"""
        X, y = self.prepare_data(series, sequence_length)
        
        if len(X) == 0:
            return
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def forecast(self, series: pd.Series, horizon: int, sequence_length: int = 12) -> np.ndarray:
        """Generate forecasts"""
        self.model.eval()
        
        # Prepare last sequence
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        
        last_sequence = scaled_values[-sequence_length:].reshape(1, sequence_length, 1)
        
        forecasts = []
        current_sequence = torch.FloatTensor(last_sequence)
        
        with torch.no_grad():
            for _ in range(horizon):
                pred = self.model(current_sequence)
                forecasts.append(pred.numpy()[0, 0])
                
                # Update sequence
                new_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(0).unsqueeze(-1)
                ], dim=1)
                current_sequence = new_sequence
        
        forecasts = np.array(forecasts).reshape(-1, 1)
        return self.scaler.inverse_transform(forecasts).flatten()
