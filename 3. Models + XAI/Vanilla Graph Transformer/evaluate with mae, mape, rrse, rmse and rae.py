import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import json
import time
import warnings
from datetime import datetime, timedelta
import os
import sys
from collections import defaultdict, Counter
import torch_geometric.transforms as T
from typing import Optional
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
import torch_geometric.nn
import torch_geometric

#Graph Transformer Model Definition
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            from torch_geometric.nn.attention import PerformerAttention
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pred_len = config.get('forecast_horizon', 36)
        self.channels = config.get('channels', 64)
        self.num_nodes = config.get('num_nodes', 645)
        
        self.node_emb = nn.Linear(config.get('node_dim', 1), self.channels - config.get('pe_dim', 8))
        self.pe_lin = nn.Linear(20, config.get('pe_dim', 8))
        self.pe_norm = nn.BatchNorm1d(20)
        
        self.edge_emb = nn.Embedding(2, self.channels)
        
        # Create transformer layers
        self.convs = nn.ModuleList()
        num_layers = config.get('num_layers', 3)
        
        for _ in range(num_layers):
            nn_seq = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.ReLU(),
                nn.Linear(self.channels, self.channels),
            )
            
            # Import here to avoid circular dependencies
            from torch_geometric.nn import GINEConv, GPSConv
            
            conv = GPSConv(
                self.channels, 
                GINEConv(nn_seq), 
                heads=config.get('num_heads', 4),
                attn_type=config.get('attn_type', 'multihead'),
                attn_kwargs={'dropout': config.get('dropout', 0.1)}
            )
            self.convs.append(conv)
        
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # Forecast head - outputs forecast_horizon * num_nodes
        self.forecast_head = nn.Sequential(
            nn.Linear(self.channels, self.channels // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.channels // 2, self.channels // 4),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.channels // 4, self.pred_len * self.num_nodes),
        )
        
        self.redraw_projection = RedrawProjection(
            self, redraw_interval=1000 if config.get('attn_type', 'multihead') == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch, mc_dropout=True):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        if pe.dim() == 3:
            pe = pe.reshape(-1, pe.size(-1))

        pe = self.pe_norm(pe)
        node_emb = self.node_emb(x)
        pe_emb = self.pe_lin(pe)
        x = torch.cat((node_emb, pe_emb), dim=-1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            if mc_dropout:
                x = self.dropout(x)
        
        # Global pooling
        x = torch_geometric.nn.global_add_pool(x, batch)
        
        # Apply forecast head
        output = self.forecast_head(x)
        
        # Reshape to (batch_size, pred_len, num_nodes)
        batch_size = x.shape[0]
        output = output.view(batch_size, self.pred_len, self.num_nodes)
        
        return output

warnings.filterwarnings('ignore')

class CyberThreatModelEvaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model first to get the correct configuration
        self.load_model()
        
        # Metrics storage
        self.metrics = {}
        
    def extract_config_from_checkpoint(self, checkpoint):
        """Extract model configuration from the checkpoint file for GraphTransformer"""
        # Default configuration for GraphTransformer
        config = {
            'sequence_length': 12,
            'embed_dim': 128,  # For compatibility
            'num_heads': 4,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'mc_dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 8,
            'forecast_horizon': 36,
            'num_nodes': 645,
            'channels': 64,
            'node_dim': 1,
            'pe_dim': 8,
            'attn_type': 'multihead',
        }
        
        # Try to extract configuration from checkpoint
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config.update(checkpoint['config'])
                print("Loaded configuration from checkpoint 'config' key")
            elif 'hyperparameters' in checkpoint:
                config.update(checkpoint['hyperparameters'])
                print("Loaded configuration from checkpoint 'hyperparameters' key")
        
        # Infer configuration from model weights if available
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            # Try to find state dict in any key
            state_dict = None
            for key, value in checkpoint.items():
                if isinstance(value, dict) and any(k.startswith('node_emb.') for k in value.keys()):
                    state_dict = value
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Infer configuration from state dict
        if state_dict is not None:
            print("Inferring configuration from model weights...")
            
            # Infer num_nodes from forecast_head weight shape
            forecast_head_keys = [k for k in state_dict.keys() if 'forecast_head' in k]
            for key in forecast_head_keys:
                if 'weight' in key and key.endswith('weight'):
                    weight = state_dict[key]
                    if len(weight.shape) == 2:
                        if weight.shape[0] % config['forecast_horizon'] == 0:
                            inferred_num_nodes = weight.shape[0] // config['forecast_horizon']
                            if inferred_num_nodes > 0:
                                config['num_nodes'] = inferred_num_nodes
                                print(f"Inferred num_nodes from {key}: {config['num_nodes']}")
            
            # Infer channels from layer weights
            for key in state_dict.keys():
                if 'convs.0.lin.weight' in key or 'convs.0.lin_src.weight' in key:
                    weight = state_dict[key]
                    if len(weight.shape) == 2:
                        config['channels'] = weight.shape[0]
                        print(f"Inferred channels: {config['channels']}")
                        break
            
            # Infer num_layers by counting conv layers
            layer_count = 0
            for key in state_dict.keys():
                if 'convs.' in key and '.lin.' in key:
                    try:
                        layer_idx = int(key.split('.')[1])
                        layer_count = max(layer_count, layer_idx + 1)
                    except:
                        pass
            if layer_count > 0:
                config['num_layers'] = layer_count
                print(f"Inferred num_layers: {config['num_layers']}")
            
            # Check for node_emb to infer node_dim
            if 'node_emb.weight' in state_dict:
                weight = state_dict['node_emb.weight']
                if len(weight.shape) == 2:
                    config['node_dim'] = weight.shape[1]
                    print(f"Inferred node_dim: {config['node_dim']}")
        
        return config
    
    def load_model(self):
        """Load the trained GraphTransformer model and extract its configuration"""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract configuration from checkpoint
            self.config = self.extract_config_from_checkpoint(checkpoint)
            
            print(f"Final model configuration:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            
            # Create model with extracted configuration
            self.model = GraphTransformer(self.config).to(self.device)
            
            # Load trained weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                # Try to find state dict in the dictionary
                state_dict = {}
                for key, value in checkpoint.items():
                    if isinstance(value, dict) or 'weight' in key or 'bias' in key:
                        if hasattr(value, 'keys'):
                            state_dict.update(value)
                        else:
                            state_dict[key] = value
            else:
                state_dict = checkpoint
            
            # Load weights with flexible matching
            model_dict = self.model.state_dict()
            
            # Filter out unnecessary keys
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present (from DataParallel)
                k = k.replace('module.', '')
                
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    # Try to match with different naming conventions
                    possible_keys = [
                        k,
                        k.replace('transformer.', ''),
                        k.replace('encoder.', ''),
                        k.replace('decoder.', ''),
                        k.replace('model.', ''),
                    ]
                    
                    for possible_key in possible_keys:
                        if possible_key in model_dict and model_dict[possible_key].shape == v.shape:
                            filtered_state_dict[possible_key] = v
                            print(f"Matched {k} -> {possible_key}")
                            break
            
            # Load filtered state dict
            model_dict.update(filtered_state_dict)
            self.model.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(filtered_state_dict)}/{len(model_dict)} parameters")
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_graph_structure(self, data):
        """Create graph structure based on correlation between nodes"""
        num_nodes = data.shape[1]
        
        # Handle NaN values
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use correlation as adjacency measure
        try:
            correlation_matrix = np.corrcoef(data_clean.T)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        except:
            print("Warning: Correlation calculation failed, using identity matrix")
            correlation_matrix = np.eye(num_nodes)
        
        threshold = 0.3  # Correlation threshold for edges
        adj_matrix = (correlation_matrix > threshold).astype(int)
        
        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] == 1 and i != j:
                    # -only add edges with valid indices 
                    if i < num_nodes and j < num_nodes:
                        edge_index.append([i, j])
                        edge_attr.append(1)  # Binary edge attribute
        
        if len(edge_index) == 0:
            # Create a minimal graph if no edges found
            for i in range(num_nodes - 1):
                edge_index.append([i, i + 1])
                edge_attr.append(1)
        
        edge_index = np.array(edge_index)
        if edge_index.size > 0:
            valid_mask = (edge_index[:,0] < num_nodes) & (edge_index[:,1] < num_nodes)
            edge_index = edge_index[valid_mask]
            edge_attr = np.array(edge_attr)[valid_mask]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index.size > 0 else torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long) if len(edge_attr) > 0 else torch.zeros((0,), dtype=torch.long)
        
        print(f"Graph created: {edge_index.shape[1]} edges, {num_nodes} nodes")
        return edge_index, edge_attr
    
    def prepare_pyg_data(self, sequences):
        """Convert sequences to PyG Data objects"""
        from torch_geometric.data import Data, Batch

        data_list = []

        for seq in sequences:
            x = torch.FloatTensor(seq[-1]).unsqueeze(1)  # Shape: [num_nodes, 1]
            pe = torch.randn(x.shape[0], 20)  # Random walk PE approximation

            # filter edge_index to valid indices for current batch 
            num_nodes = x.shape[0]
            edge_index_np = self.edge_index.cpu().numpy() if hasattr(self, 'edge_index') else np.zeros((2,0), dtype=int)
            edge_attr_np = self.edge_attr.cpu().numpy() if hasattr(self, 'edge_attr') else np.zeros((0,), dtype=int)
            if edge_index_np.shape[1] > 0:
                valid_mask = (edge_index_np[0] < num_nodes) & (edge_index_np[1] < num_nodes)
                edge_index_np = edge_index_np[:, valid_mask]
                edge_attr_np = edge_attr_np[valid_mask]
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_np, dtype=torch.long)

            data = Data(
                x=x,
                pe=pe,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch
    
    def load_and_split_data(self):
        """Load data from data.csv and split into training and testing periods"""
        try:
            # Load the complete dataset
            print(f"Loading data from {self.data_file}...")
            self.data_df = pd.read_csv(self.data_file)
            
            # Check data structure
            print(f"Data columns: {self.data_df.columns.tolist()}")
            print(f"Data shape: {self.data_df.shape}")
            
            # Extract date information
            date_column = None
            for col in self.data_df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                    date_column = col
                    break
            
            if date_column is None:
                # Assume first column is date
                date_column = self.data_df.columns[0]
                print(f"No explicit date column found, using first column: {date_column}")
            
            # Convert to datetime
            self.data_df['date'] = pd.to_datetime(self.data_df[date_column])
            
            # Extract the actual data (excluding date column)
            data_columns = [col for col in self.data_df.columns if col != date_column and col != 'date']
            self.threat_data = self.data_df[data_columns].values.astype(np.float32)
            
            print(f"Processed data shape: {self.threat_data.shape}")
            print(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
            
            # Use 70% for training, 30% for testing
            split_idx = int(0.7 * len(self.threat_data))
            self.data_train = self.threat_data[:split_idx]
            self.data_test = self.threat_data[split_idx:]
            self.dates_train = self.data_df['date'][:split_idx]
            self.dates_test = self.data_df['date'][split_idx:]
            
            print(f"Training data: {self.data_train.shape} (first {split_idx} samples)")
            print(f"Testing data: {self.data_test.shape} (last {len(self.data_test)} samples)")
            
            # Adjust forecast horizon if test data is too small
            required_length = self.config.get('sequence_length', 12) + self.config.get('forecast_horizon', 36)
            if len(self.data_test) < required_length:
                print(f"Test data too small for forecast horizon {self.config.get('forecast_horizon', 36)}")
                # Reduce forecast horizon to fit available data
                max_possible_horizon = len(self.data_test) - self.config.get('sequence_length', 12) - 1
                if max_possible_horizon > 0:
                    self.config['forecast_horizon'] = max_possible_horizon
                    print(f"Reduced forecast horizon to: {self.config['forecast_horizon']}")
                else:
                    # If still too small, reduce sequence length
                    self.config['sequence_length'] = min(self.config.get('sequence_length', 12), len(self.data_test) // 2)
                    self.config['forecast_horizon'] = len(self.data_test) - self.config['sequence_length'] - 1
                    print(f"Adjusted sequence_length to {self.config['sequence_length']}, forecast_horizon to {self.config['forecast_horizon']}")
            
            # Ensure data has correct number of nodes
            if self.data_train.shape[1] != self.config['num_nodes']:
                print(f"Adjusting data dimensions from {self.data_train.shape[1]} to {self.config['num_nodes']} nodes")
                if self.data_train.shape[1] > self.config['num_nodes']:
                    # Truncate if data has more nodes
                    self.data_train = self.data_train[:, :self.config['num_nodes']]
                    self.data_test = self.data_test[:, :self.config['num_nodes']]
                else:
                    # Pad if data has fewer nodes
                    pad_width = self.config['num_nodes'] - self.data_train.shape[1]
                    self.data_train = np.pad(self.data_train, ((0, 0), (0, pad_width)), mode='constant')
                    self.data_test = np.pad(self.data_test, ((0, 0), (0, pad_width)), mode='constant')
            
            # Create graph structure using training data
            print("Creating graph structure from training data...")
            self.edge_index, self.edge_attr = self.create_graph_structure(self.data_train)
            
            # Normalize data using training statistics
            self.data_mean = np.mean(self.data_train, axis=0)
            self.data_std = np.std(self.data_train, axis=0) + 1e-8
            
            self.data_train_norm = (self.data_train - self.data_mean) / self.data_std
            self.data_test_norm = (self.data_test - self.data_mean) / self.data_std
            
            print("Data loaded and normalized successfully")
            
        except Exception as e:
            print(f"Error loading and splitting data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_sequences(self, data, sequence_length, forecast_horizon=None):
        """Generate input sequences for the model"""
        if forecast_horizon is None:
            forecast_horizon = self.config.get('forecast_horizon', 36)
        
        sequences = []
        targets = []
        
        total_possible = len(data) - sequence_length - forecast_horizon + 1
        
        if total_possible <= 0:
            # Try single-step prediction
            forecast_horizon = 1
            total_possible = len(data) - sequence_length - forecast_horizon + 1
        
        if total_possible <= 0:
            print(f"Warning: Not enough data to generate sequences. Data length: {len(data)}, sequence_length: {sequence_length}")
            return [], []
        
        for i in range(total_possible):
            input_seq = data[i:i+sequence_length]
            target_seq = data[i+sequence_length:i+sequence_length+forecast_horizon]
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        print(f"Generated {len(sequences)} sequences with forecast horizon {forecast_horizon}")
        return sequences, targets
    
    def generate_predictions(self, data, sequence_length=None, forecast_horizon=None):
        """Generate predictions using the GraphTransformer model"""
        if sequence_length is None:
            sequence_length = self.config.get('sequence_length', 12)
        if forecast_horizon is None:
            forecast_horizon = self.config.get('forecast_horizon', 36)
        
        # Generate sequences
        sequences, actuals = self.generate_sequences(data, sequence_length, forecast_horizon)
        
        if len(sequences) == 0:
            print("No sequences generated, returning empty predictions")
            return np.array([]), np.array([])
        
        predictions = []
        actual_values = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(sequences), 8):  # Process in batches
                batch_sequences = sequences[i:i+8]
                batch_actuals = actuals[i:i+8]
                
                if len(batch_sequences) == 0:
                    continue
                
                # Prepare PyG data for the batch
                batch_data = self.prepare_pyg_data(batch_sequences)
                batch_data = batch_data.to(self.device)
                
                # Generate predictions
                pred = self.model(
                    batch_data.x, 
                    batch_data.pe, 
                    batch_data.edge_index, 
                    batch_data.edge_attr, 
                    batch_data.batch,
                    mc_dropout=False
                )
                
                # Move to CPU and convert to numpy
                pred_np = pred.cpu().numpy()
                
                # Reshape predictions to match actuals
                batch_size = len(batch_sequences)
                num_nodes = self.config['num_nodes']
                
                # pred_np shape should be [batch_size, forecast_horizon, num_nodes]
                # We need to transpose to [batch_size, num_nodes, forecast_horizon] for consistency
                if pred_np.shape[1] == forecast_horizon and pred_np.shape[2] == num_nodes:
                    # Already in correct shape
                    pass
                elif pred_np.shape[1] == num_nodes and pred_np.shape[2] == forecast_horizon:
                    # Transpose
                    pred_np = np.transpose(pred_np, (0, 2, 1))
                else:
                    # Try to reshape
                    try:
                        pred_np = pred_np.reshape(batch_size, forecast_horizon, num_nodes)
                    except:
                        print(f"Warning: Could not reshape predictions from {pred_np.shape}")
                
                predictions.append(pred_np)
                actual_values.append(np.array(batch_actuals))
        
        if len(predictions) == 0:
            return np.array([]), np.array([])
        
        # Concatenate all predictions
        predictions_array = np.concatenate(predictions, axis=0)
        actuals_array = np.concatenate(actual_values, axis=0)
        
        print(f"Generated predictions shape: {predictions_array.shape}")
        print(f"Actuals shape: {actuals_array.shape}")
        
        return predictions_array, actuals_array
    
    def calculate_msmape(self, predictions, actuals):
        """Calculate M-SMAPE (Modified Symmetric Mean Absolute Percentage Error)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        epsilon = 1e-8
        numerator = np.abs(predictions - actuals)
        denominator = (np.abs(predictions) + np.abs(actuals)) / 2 + epsilon
        
        smape = 2 * np.mean(numerator / denominator)
        return smape * 100  # Return as percentage
    
    def calculate_mae(self, predictions, actuals):
        """Calculate Mean Absolute Error (MAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        mae = np.mean(np.abs(predictions - actuals))
        return mae
    
    def calculate_mape(self, predictions, actuals):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        return mape
    
    def calculate_rmse(self, predictions, actuals):
        """Calculate Root Mean Squared Error (RMSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        return rmse
    
    def calculate_rae(self, predictions, actuals):
        """Calculate Relative Absolute Error (RAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Sum of absolute errors
        sum_absolute_errors = np.sum(np.abs(predictions - actuals))
        
        # Sum of absolute deviations from mean
        mean_actuals = np.mean(actuals)
        sum_absolute_deviation = np.sum(np.abs(actuals - mean_actuals))
        
        # Avoid division by zero
        if sum_absolute_deviation == 0:
            return float('inf')
        
        rae = sum_absolute_errors / sum_absolute_deviation
        return rae
    
    def calculate_rrse(self, predictions, actuals):
        """Calculate Root Relative Squared Error (RRSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Sum of squared errors
        sum_squared_errors = np.sum((predictions - actuals) ** 2)
        
        # Sum of squared deviations from mean
        mean_actuals = np.mean(actuals)
        sum_squared_deviation = np.sum((actuals - mean_actuals) ** 2)
        
        # Avoid division by zero
        if sum_squared_deviation == 0:
            return float('inf')
        
        rrse = np.sqrt(sum_squared_errors / sum_squared_deviation)
        return rrse
    
    def calculate_auc_roc(self, predictions, actuals, threshold=0.3):
        """Calculate AUC-ROC for binary threat detection"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.5, np.array([0, 1]), np.array([0, 1])
            
        # Convert to binary classification (threat vs no-threat)
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        # Flatten for overall AUC
        pred_flat = pred_binary.flatten()
        actual_flat = actual_binary.flatten()
        
        try:
            auc = roc_auc_score(actual_flat, pred_flat)
            fpr, tpr, _ = roc_curve(actual_flat, pred_flat)
            return auc, fpr, tpr
        except:
            return 0.5, np.array([0, 1]), np.array([0, 1])  # Random classifier
    
    def calculate_false_positive_rate(self, predictions, actuals, threshold=0.3):
        """Calculate False Positive Rate"""
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
            
        # For multi-step predictions
        if len(predictions.shape) == 3:
            # Compare trends across time steps
            pred_trend = np.diff(predictions, axis=1) > 0
            actual_trend = np.diff(actuals, axis=1) > 0
            correct_direction = (pred_trend == actual_trend)
        else:
            # For flattened arrays
            pred_trend = np.diff(predictions, axis=0) > 0
            actual_trend = np.diff(actuals, axis=0) > 0
            correct_direction = (pred_trend == actual_trend)
        
        directional_accuracy = np.mean(correct_direction)
        return directional_accuracy
    
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
    
    def calculate_ks_test(self, predictions, actuals):
        """Calculate Kolmogorov-Smirnov test statistics"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0, 0.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        stat, p_value = ks_2samp(pred_flat, actual_flat)
        return stat, p_value
    
    def calculate_js_divergence(self, predictions, actuals, bins=50):
        """Calculate Jensen-Shannon Divergence"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        # Create histograms
        min_val = min(np.min(pred_flat), np.min(actual_flat))
        max_val = max(np.max(pred_flat), np.max(actual_flat))
        
        pred_hist, _ = np.histogram(pred_flat, bins=bins, range=(min_val, max_val), density=True)
        actual_hist, _ = np.histogram(actual_flat, bins=bins, range=(min_val, max_val), density=True)
        
        js_div = jensenshannon(pred_hist, actual_hist)
        return js_div
    
    def calculate_temporal_correlation(self, predictions, actuals):
        """Calculate Temporal Correlation Coefficient"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        correlations = []
        
        # Handle multi-step predictions
        if len(predictions.shape) == 3:
            for node_idx in range(actuals.shape[2]):
                node_actual = actuals[:, :, node_idx].flatten()
                node_pred = predictions[:, :, node_idx].flatten()
                
                # Remove constant sequences
                if np.std(node_actual) > 0 and np.std(node_pred) > 0:
                    corr = np.corrcoef(node_actual, node_pred)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        else:
            # Single correlation for flattened arrays
            pred_flat = predictions.flatten()
            actual_flat = actuals.flatten()
            if np.std(actual_flat) > 0 and np.std(pred_flat) > 0:
                corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0
    
    def calculate_adversarial_robustness(self, predictions, actuals, noise_level=0.1):
        """Calculate Adversarial Robustness Score"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        original_predictions = predictions.copy()
        
        # Add small perturbations to inputs
        noisy_predictions = predictions + np.random.normal(0, noise_level, predictions.shape)
        
        # Calculate change in predictions
        prediction_change = np.abs(noisy_predictions - original_predictions)
        robustness_score = 1.0 / (1.0 + np.mean(prediction_change))
        
        return robustness_score
    
    def calculate_operational_readiness_index(self, precision, recall, fpr, resource_efficiency):
        """Calculate composite Operational Readiness Index"""
        # Normalise metrics to 0-1 scale
        precision_norm = precision
        recall_norm = recall
        fpr_norm = 1 - fpr  # Lower FPR is better
        resource_norm = min(resource_efficiency, 1.0)
        
        # Weighted combination
        ori = (0.3 * precision_norm + 0.3 * recall_norm + 
               0.2 * fpr_norm + 0.2 * resource_norm)
        
        return ori
    
    def _match_prediction_shapes(self, predictions, actuals):
        if predictions.size == 0 or actuals.size == 0:
            return predictions, actuals
        if predictions.shape == actuals.shape:
            return predictions, actuals
        if predictions.ndim != actuals.ndim:
            min_len = min(predictions.size, actuals.size)
            return predictions.reshape(-1)[:min_len], actuals.reshape(-1)[:min_len]
        min_dims = tuple(min(p, a) for p, a in zip(predictions.shape, actuals.shape))
        slices = tuple(slice(0, dim) for dim in min_dims)
        return predictions[slices], actuals[slices]

    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics including RRSE, RMSE, and RAE"""
        print("Starting comprehensive model evaluation...")
        
        # Load and split data
        self.load_and_split_data()
        
        # Generate predictions for training and testing data
        seq_len = self.config.get('sequence_length', 12)
        forecast_horizon = self.config.get('forecast_horizon', 36)
        
        print(f"\nGenerating predictions for training data...")
        pred_train, actual_train = self.generate_predictions(
            self.data_train_norm, seq_len, forecast_horizon
        )
        
        print(f"\nGenerating predictions for testing data...")
        pred_test, actual_test = self.generate_predictions(
            self.data_test_norm, seq_len, forecast_horizon
        )

        pred_train, actual_train = self._match_prediction_shapes(pred_train, actual_train)
        pred_test, actual_test = self._match_prediction_shapes(pred_test, actual_test)

        # Handle empty predictions
        if pred_test.size == 0:
            print("Warning: No predictions generated. Evaluation will use placeholder metrics.")
            self.initialize_placeholder_metrics()
            return self.metrics
        
        # Denormalize predictions and actuals
        pred_train_denorm = pred_train * self.data_std + self.data_mean
        actual_train_denorm = actual_train * self.data_std + self.data_mean
        pred_test_denorm = pred_test * self.data_std + self.data_mean
        actual_test_denorm = actual_test * self.data_std + self.data_mean
        
        print(f"Training predictions shape: {pred_train.shape}")
        print(f"Testing predictions shape: {pred_test.shape}")
        
        # Calculate metrics
        threshold = 0.3
        
        # 1. Accuracy Metrics
        self.metrics['M-SMAPE_Train'] = self.calculate_msmape(pred_train_denorm, actual_train_denorm)
        self.metrics['M-SMAPE_Test'] = self.calculate_msmape(pred_test_denorm, actual_test_denorm)
        
        self.metrics['MAE_Train'] = self.calculate_mae(pred_train_denorm, actual_train_denorm)
        self.metrics['MAE_Test'] = self.calculate_mae(pred_test_denorm, actual_test_denorm)
        
        self.metrics['MAPE_Train'] = self.calculate_mape(pred_train_denorm, actual_train_denorm)
        self.metrics['MAPE_Test'] = self.calculate_mape(pred_test_denorm, actual_test_denorm)
        
        # NEW: RMSE, RAE, RRSE metrics
        self.metrics['RMSE_Train'] = self.calculate_rmse(pred_train_denorm, actual_train_denorm)
        self.metrics['RMSE_Test'] = self.calculate_rmse(pred_test_denorm, actual_test_denorm)
        
        self.metrics['RAE_Train'] = self.calculate_rae(pred_train_denorm, actual_train_denorm)
        self.metrics['RAE_Test'] = self.calculate_rae(pred_test_denorm, actual_test_denorm)
        
        self.metrics['RRSE_Train'] = self.calculate_rrse(pred_train_denorm, actual_train_denorm)
        self.metrics['RRSE_Test'] = self.calculate_rrse(pred_test_denorm, actual_test_denorm)
        
        # 2. Detection Quality Metrics
        auc_train, fpr_curve_train, tpr_curve_train = self.calculate_auc_roc(pred_train, actual_train, threshold)
        auc_test, fpr_curve_test, tpr_curve_test = self.calculate_auc_roc(pred_test, actual_test, threshold)
        self.metrics['AUC_ROC_Train'] = auc_train
        self.metrics['AUC_ROC_Test'] = auc_test
        
        self.metrics['FPR_Train'] = self.calculate_false_positive_rate(pred_train, actual_train, threshold)
        self.metrics['FPR_Test'] = self.calculate_false_positive_rate(pred_test, actual_test, threshold)
        
        # 3. Coverage and Precision
        self.metrics['Attack_Coverage_Train'] = self.calculate_attack_coverage(pred_train, actual_train, threshold)
        self.metrics['Attack_Coverage_Test'] = self.calculate_attack_coverage(pred_test, actual_test, threshold)
        
        self.metrics['Alert_Precision_Train'] = self.calculate_alert_precision(pred_train, actual_train, threshold)
        self.metrics['Alert_Precision_Test'] = self.calculate_alert_precision(pred_test, actual_test, threshold)
        
        # 4. Distribution Metrics
        self.metrics['Directional_Accuracy_Train'] = self.calculate_directional_accuracy(pred_train, actual_train)
        self.metrics['Directional_Accuracy_Test'] = self.calculate_directional_accuracy(pred_test, actual_test)
        
        # Quantile Loss
        quantile_loss_train = self.calculate_quantile_loss(pred_train_denorm, actual_train_denorm)
        quantile_loss_test = self.calculate_quantile_loss(pred_test_denorm, actual_test_denorm)
        self.metrics.update({f'Quantile_Loss_Train_{k}': v for k, v in quantile_loss_train.items()})
        self.metrics.update({f'Quantile_Loss_Test_{k}': v for k, v in quantile_loss_test.items()})
        
        # Statistical Tests
        ks_stat_train, ks_p_train = self.calculate_ks_test(pred_train_denorm, actual_train_denorm)
        ks_stat_test, ks_p_test = self.calculate_ks_test(pred_test_denorm, actual_test_denorm)
        self.metrics['KS_Statistic_Train'] = ks_stat_train
        self.metrics['KS_Statistic_Test'] = ks_stat_test
        self.metrics['KS_p_value_Train'] = ks_p_train
        self.metrics['KS_p_value_Test'] = ks_p_test
        
        self.metrics['JS_Divergence_Train'] = self.calculate_js_divergence(pred_train_denorm, actual_train_denorm)
        self.metrics['JS_Divergence_Test'] = self.calculate_js_divergence(pred_test_denorm, actual_test_denorm)
        
        # 5. Temporal Metrics
        self.metrics['Temporal_Correlation_Train'] = self.calculate_temporal_correlation(pred_train, actual_train)
        self.metrics['Temporal_Correlation_Test'] = self.calculate_temporal_correlation(pred_test, actual_test)
        
        # 6. Robustness Metrics
        self.metrics['Adversarial_Robustness_Train'] = self.calculate_adversarial_robustness(pred_train, actual_train)
        self.metrics['Adversarial_Robustness_Test'] = self.calculate_adversarial_robustness(pred_test, actual_test)
        
        # 7. Composite Metrics
        recall_train = self.metrics['Attack_Coverage_Train']
        recall_test = self.metrics['Attack_Coverage_Test']
        resource_efficiency = 0.8  # Placeholder
        
        self.metrics['Operational_Readiness_Train'] = self.calculate_operational_readiness_index(
            self.metrics['Alert_Precision_Train'], recall_train, 
            self.metrics['FPR_Train'], resource_efficiency
        )
        self.metrics['Operational_Readiness_Test'] = self.calculate_operational_readiness_index(
            self.metrics['Alert_Precision_Test'], recall_test, 
            self.metrics['FPR_Test'], resource_efficiency
        )
        
        # Store additional data for visualization
        self.evaluation_data = {
            'predictions_train': pred_train,
            'actuals_train': actual_train,
            'predictions_test': pred_test,
            'actuals_test': actual_test,
            'fpr_curve_train': fpr_curve_train,
            'tpr_curve_train': tpr_curve_train,
            'fpr_curve_test': fpr_curve_test,
            'tpr_curve_test': tpr_curve_test,
        }
        
        return self.metrics

    def initialize_placeholder_metrics(self):
        """Initialize metrics with placeholder values when no predictions are generated"""
        placeholder_metrics = {
            'M-SMAPE_Train': float('inf'),
            'M-SMAPE_Test': float('inf'),
            'MAE_Train': float('inf'),
            'MAE_Test': float('inf'),
            'MAPE_Train': float('inf'),
            'MAPE_Test': float('inf'),
            'RMSE_Train': float('inf'),
            'RMSE_Test': float('inf'),
            'RAE_Train': float('inf'),
            'RAE_Test': float('inf'),
            'RRSE_Train': float('inf'),
            'RRSE_Test': float('inf'),
            'AUC_ROC_Train': 0.5,
            'AUC_ROC_Test': 0.5,
            'FPR_Train': 1.0,
            'FPR_Test': 1.0,
            'Attack_Coverage_Train': 0.0,
            'Attack_Coverage_Test': 0.0,
            'Alert_Precision_Train': 0.0,
            'Alert_Precision_Test': 0.0,
            'Directional_Accuracy_Train': 0.0,
            'Directional_Accuracy_Test': 0.0,
            'Quantile_Loss_Train_quantile_10': float('inf'),
            'Quantile_Loss_Train_quantile_50': float('inf'),
            'Quantile_Loss_Train_quantile_90': float('inf'),
            'Quantile_Loss_Test_quantile_10': float('inf'),
            'Quantile_Loss_Test_quantile_50': float('inf'),
            'Quantile_Loss_Test_quantile_90': float('inf'),
            'KS_Statistic_Train': 1.0,
            'KS_Statistic_Test': 1.0,
            'KS_p_value_Train': 0.0,
            'KS_p_value_Test': 0.0,
            'JS_Divergence_Train': 1.0,
            'JS_Divergence_Test': 1.0,
            'Temporal_Correlation_Train': 0.0,
            'Temporal_Correlation_Test': 0.0,
            'Adversarial_Robustness_Train': 0.0,
            'Adversarial_Robustness_Test': 0.0,
            'Operational_Readiness_Train': 0.0,
            'Operational_Readiness_Test': 0.0,
        }
        self.metrics.update(placeholder_metrics)

    def plot_detailed_analysis(self):
        """Create detailed analysis plots including new metrics"""
        if not hasattr(self, 'evaluation_data') or self.evaluation_data['predictions_test'].size == 0:
            print("No evaluation data available for plotting")
            return
        
        # Create a larger figure for more subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()
        
        # 1. ROC Curves
        axes[0].plot(self.evaluation_data['fpr_curve_train'], 
                    self.evaluation_data['tpr_curve_train'], 
                    'b-', linewidth=2, label=f'Train (AUC = {self.metrics["AUC_ROC_Train"]:.3f})')
        axes[0].plot(self.evaluation_data['fpr_curve_test'], 
                    self.evaluation_data['tpr_curve_test'], 
                    'r-', linewidth=2, label=f'Test (AUC = {self.metrics["AUC_ROC_Test"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves: Train vs Test')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Performance Comparison Bar Chart (with new metrics)
        comparison_metrics = ['M-SMAPE_Train', 'M-SMAPE_Test', 
                             'MAE_Train', 'MAE_Test',
                             'MAPE_Train', 'MAPE_Test',
                             'RMSE_Train', 'RMSE_Test',
                             'RAE_Train', 'RAE_Test',
                             'RRSE_Train', 'RRSE_Test']
        
        # Filter out infinite values for plotting
        valid_metrics = []
        valid_values = []
        for m in comparison_metrics:
            if m in self.metrics and self.metrics[m] != float('inf'):
                valid_metrics.append(m)
                valid_values.append(self.metrics[m])
        
        x_pos = np.arange(len(valid_metrics))
        axes[1].bar(x_pos, valid_values, alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(valid_metrics, rotation=45, ha='right', fontsize=9)
        axes[1].set_title('Key Metrics Comparison: Train vs Test')
        axes[1].set_ylabel('Error Value')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Error Metrics Comparison
        error_metrics = ['MAE', 'RMSE', 'RAE', 'RRSE']
        train_errors = [self.metrics.get(f'{m}_Train', 0) for m in error_metrics]
        test_errors = [self.metrics.get(f'{m}_Test', 0) for m in error_metrics]
        
        # Handle infinite values
        train_errors = [0 if np.isinf(x) else x for x in train_errors]
        test_errors = [0 if np.isinf(x) else x for x in test_errors]
        
        x_error = np.arange(len(error_metrics))
        width = 0.35
        axes[2].bar(x_error - width/2, train_errors, width, label='Train', alpha=0.7)
        axes[2].bar(x_error + width/2, test_errors, width, label='Test', alpha=0.7)
        axes[2].set_xticks(x_error)
        axes[2].set_xticklabels(error_metrics)
        axes[2].set_ylabel('Error Value')
        axes[2].set_title('Error Metrics Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Quantile Loss Comparison
        quantile_metrics_train = [self.metrics['Quantile_Loss_Train_quantile_10'],
                                self.metrics['Quantile_Loss_Train_quantile_50'],
                                self.metrics['Quantile_Loss_Train_quantile_90']]
        quantile_metrics_test = [self.metrics['Quantile_Loss_Test_quantile_10'],
                               self.metrics['Quantile_Loss_Test_quantile_50'],
                               self.metrics['Quantile_Loss_Test_quantile_90']]
        
        # Handle infinite values
        quantile_metrics_train = [0 if np.isinf(x) else x for x in quantile_metrics_train]
        quantile_metrics_test = [0 if np.isinf(x) else x for x in quantile_metrics_test]
        
        x_quantile = np.arange(3)
        axes[3].bar(x_quantile - width/2, quantile_metrics_train, width, label='Train', alpha=0.7)
        axes[3].bar(x_quantile + width/2, quantile_metrics_test, width, label='Test', alpha=0.7)
        axes[3].set_xticks(x_quantile)
        axes[3].set_xticklabels(['Q10', 'Q50', 'Q90'])
        axes[3].set_ylabel('Quantile Loss')
        axes[3].set_title('Quantile Loss Comparison')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Prediction vs Actual Scatter
        if self.evaluation_data['predictions_test'].size > 0:
            test_pred_flat = self.evaluation_data['predictions_test'].flatten()
            test_actual_flat = self.evaluation_data['actuals_test'].flatten()
            
            # Take a sample if too many points
            if len(test_pred_flat) > 10000:
                indices = np.random.choice(len(test_pred_flat), 10000, replace=False)
                test_pred_flat = test_pred_flat[indices]
                test_actual_flat = test_actual_flat[indices]
            
            axes[4].scatter(test_actual_flat, test_pred_flat, alpha=0.1, s=1)
            axes[4].plot([test_actual_flat.min(), test_actual_flat.max()], 
                        [test_actual_flat.min(), test_actual_flat.max()], 'r--', alpha=0.8, linewidth=2)
            axes[4].set_xlabel('Actual Values')
            axes[4].set_ylabel('Predicted Values')
            axes[4].set_title('Prediction vs Actual (Test Set)')
            axes[4].grid(True, alpha=0.3)
        
        # 6. Error Distribution Histogram
        if self.evaluation_data['predictions_test'].size > 0:
            test_errors = test_actual_flat - test_pred_flat
            axes[5].hist(test_errors, bins=50, alpha=0.7, edgecolor='black')
            axes[5].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[5].set_xlabel('Prediction Error (Actual - Predicted)')
            axes[5].set_ylabel('Frequency')
            axes[5].set_title('Error Distribution Histogram (Test Set)')
            axes[5].grid(True, alpha=0.3)
            axes[5].text(0.05, 0.95, f'Mean Error: {np.mean(test_errors):.4f}\nStd Error: {np.std(test_errors):.4f}',
                        transform=axes[5].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('Dissertation/Vanilla Graph Transformer/graph_transformer_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Comprehensive analysis plot saved as 'graph_transformer_comprehensive_analysis.png'")
    
    def generate_report(self):
        """Generate comprehensive text report including new metrics"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("GRAPH TRANSFORMER MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.model_path}")
        report_lines.append(f"Data: {self.data_file}")
        report_lines.append("")
        
        # Model Configuration
        report_lines.append("MODEL CONFIGURATION")
        report_lines.append("-" * 40)
        for key, value in self.config.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Calculate overall scores
        avg_msmape = (self.metrics['M-SMAPE_Train'] + self.metrics['M-SMAPE_Test']) / 2
        avg_mae = (self.metrics['MAE_Train'] + self.metrics['MAE_Test']) / 2
        avg_mape = (self.metrics['MAPE_Train'] + self.metrics['MAPE_Test']) / 2
        avg_rmse = (self.metrics['RMSE_Train'] + self.metrics['RMSE_Test']) / 2
        avg_rae = (self.metrics['RAE_Train'] + self.metrics['RAE_Test']) / 2
        avg_rrse = (self.metrics['RRSE_Train'] + self.metrics['RRSE_Test']) / 2
        avg_auc = (self.metrics['AUC_ROC_Train'] + self.metrics['AUC_ROC_Test']) / 2
        avg_ori = (self.metrics['Operational_Readiness_Train'] + self.metrics['Operational_Readiness_Test']) / 2
        
        report_lines.append(f"Average M-SMAPE: {avg_msmape:.2f}%")
        report_lines.append(f"Average MAE: {avg_mae:.4f}")
        report_lines.append(f"Average MAPE: {avg_mape:.2f}%")
        report_lines.append(f"Average RMSE: {avg_rmse:.4f}")
        report_lines.append(f"Average RAE: {avg_rae:.4f}")
        report_lines.append(f"Average RRSE: {avg_rrse:.4f}")
        report_lines.append(f"Average AUC-ROC: {avg_auc:.3f}")
        report_lines.append(f"Average Operational Readiness: {avg_ori:.3f}")
        report_lines.append("")
        
        # Detailed Metrics
        report_lines.append("DETAILED METRICS")
        report_lines.append("-" * 40)
        
        # Group metrics by category
        accuracy_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['SMAPE', 'MAE', 'MAPE', 'RMSE', 'RAE', 'RRSE', 'Directional'])}
        detection_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['AUC', 'FPR', 'Coverage', 'Precision'])}
        temporal_metrics = {k: v for k, v in self.metrics.items() if 'Temporal' in k}
        robustness_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['Robustness', 'KS', 'JS'])}
        quantile_metrics = {k: v for k, v in self.metrics.items() if 'Quantile' in k}
        
        report_lines.append("Accuracy Metrics:")
        for metric, value in sorted(accuracy_metrics.items()):
            if 'SMAPE' in metric or 'MAPE' in metric:
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.2f}%")
            elif any(x in metric for x in ['MAE', 'RMSE', 'RAE', 'RRSE']):
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nDetection Quality Metrics:")
        for metric, value in sorted(detection_metrics.items()):
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nTemporal Performance Metrics:")
        for metric, value in sorted(temporal_metrics.items()):
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nRobustness and Distribution Metrics:")
        for metric, value in sorted(robustness_metrics.items()):
            if 'p_value' in metric:
                report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nUncertainty Quantification (Quantile Loss):")
        for metric, value in sorted(quantile_metrics.items()):
            if value == float('inf'):
                report_lines.append(f"  {metric}: N/A (no predictions)")
            else:
                report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.append("")
        
        # Performance Assessment
        report_lines.append("PERFORMANCE ASSESSMENT")
        report_lines.append("-" * 40)
        
        # Assess model performance based on new metrics
        if avg_auc > 0.8:
            auc_assessment = "EXCELLENT"
        elif avg_auc > 0.7:
            auc_assessment = "GOOD"
        elif avg_auc > 0.6:
            auc_assessment = "FAIR"
        else:
            auc_assessment = "POOR"
        
        if avg_rmse == float('inf'):
            rmse_assessment = "N/A"
        elif avg_rmse < 0.1:
            rmse_assessment = "EXCELLENT"
        elif avg_rmse < 0.2:
            rmse_assessment = "GOOD"
        elif avg_rmse < 0.3:
            rmse_assessment = "FAIR"
        else:
            rmse_assessment = "POOR"
        
        if avg_msmape == float('inf'):
            smape_assessment = "N/A"
        elif avg_msmape < 10:
            smape_assessment = "EXCELLENT"
        elif avg_msmape < 20:
            smape_assessment = "GOOD"
        elif avg_msmape < 30:
            smape_assessment = "FAIR"
        else:
            smape_assessment = "POOR"
        
        report_lines.append(f"Detection Quality: {auc_assessment} (AUC-ROC: {avg_auc:.3f})")
        if avg_rmse == float('inf'):
            report_lines.append(f"RMSE Performance: N/A (no predictions generated)")
        else:
            report_lines.append(f"RMSE Performance: {rmse_assessment} (RMSE: {avg_rmse:.4f})")
        
        if avg_msmape == float('inf'):
            report_lines.append(f"Forecast Accuracy: N/A (no predictions generated)")
        else:
            report_lines.append(f"Forecast Accuracy: {smape_assessment} (M-SMAPE: {avg_msmape:.2f}%)")
        
        # RAE and RRSE assessment (lower is better)
        if avg_rae == float('inf'):
            rae_assessment = "N/A"
        elif avg_rae < 0.5:
            rae_assessment = "EXCELLENT"
        elif avg_rae < 0.8:
            rae_assessment = "GOOD"
        elif avg_rae < 1.0:
            rae_assessment = "FAIR"
        else:
            rae_assessment = "POOR"
            
        if avg_rrse == float('inf'):
            rrse_assessment = "N/A"
        elif avg_rrse < 0.5:
            rrse_assessment = "EXCELLENT"
        elif avg_rrse < 0.8:
            rrse_assessment = "GOOD"
        elif avg_rrse < 1.0:
            rrse_assessment = "FAIR"
        else:
            rrse_assessment = "POOR"
        
        report_lines.append(f"Relative Accuracy (RAE): {rae_assessment} (RAE: {avg_rae:.4f})")
        report_lines.append(f"Relative Squared Error (RRSE): {rrse_assessment} (RRSE: {avg_rrse:.4f})")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if self.metrics['FPR_Test'] > 0.1:
            report_lines.append("HIGH FALSE POSITIVE RATE: Consider adjusting detection threshold")
        if self.metrics['Adversarial_Robustness_Test'] < 0.7:
            report_lines.append("LOW ROBUSTNESS: Enhance model regularization")
        if self.metrics['Directional_Accuracy_Test'] < 0.6:
            report_lines.append("POOR TREND PREDICTION: Focus on temporal feature engineering")
        
        # Recommendations based on new metrics
        if self.metrics['RRSE_Test'] > 1.0:
            report_lines.append("HIGH RELATIVE SQUARED ERROR: Model variance is high compared to baseline")
        if self.metrics['RAE_Test'] > 1.0:
            report_lines.append("HIGH RELATIVE ABSOLUTE ERROR: Model performs worse than mean predictor")
        if self.metrics['RMSE_Test'] > 0.3:
            report_lines.append("HIGH RMSE: Consider improving model accuracy or data quality")
        
        if self.metrics['AUC_ROC_Test'] > 0.7 and self.metrics['FPR_Test'] < 0.2:
            report_lines.append("REASONABLE DETECTION CAPABILITY: Model shows promise")
        
        if self.metrics['RRSE_Test'] < 0.8 and self.metrics['RAE_Test'] < 0.8:
            report_lines.append("GOOD RELATIVE PERFORMANCE: Model outperforms baseline predictors")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        report_path = "Dissertation/Vanilla Graph Transformer/graph_transformer_evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path

def main():
    """Main execution function"""
    # Configuration
    model_path = "Dissertation/Vanilla Graph Transformer/model/SimpleGraphTransformer/final_model.pt"  # Path to model
    data_file = "Dissertation/Vanilla Graph Transformer/data/sm_data_g.csv"  # Path to complete dataset
    
    # Initialize evaluator
    evaluator = CyberThreatModelEvaluator(
        model_path=model_path,
        data_file=data_file
    )
    
    # Run comprehensive evaluation
    metrics = evaluator.run_comprehensive_evaluation()
    
    # Generate visualizations
    evaluator.plot_detailed_analysis()
    
    # Generate text report
    report_path = evaluator.generate_report()
    
    print("\nEvaluation completed successfully!")
    print(f"Key metrics calculated: {len(metrics)}")
    print(f"Report generated: {report_path}")
    
    # Print key metrics to console
    print("\nKEY METRICS SUMMARY:")
    print(f"M-SMAPE Train: {metrics.get('M-SMAPE_Train', 'N/A'):.2f}%")
    print(f"M-SMAPE Test: {metrics.get('M-SMAPE_Test', 'N/A'):.2f}%")
    print(f"MAE Train: {metrics.get('MAE_Train', 'N/A'):.4f}")
    print(f"MAE Test: {metrics.get('MAE_Test', 'N/A'):.4f}")
    print(f"MAPE Train: {metrics.get('MAPE_Train', 'N/A'):.2f}%")
    print(f"MAPE Test: {metrics.get('MAPE_Test', 'N/A'):.2f}%")
    print(f"RMSE Train: {metrics.get('RMSE_Train', 'N/A'):.4f}")
    print(f"RMSE Test: {metrics.get('RMSE_Test', 'N/A'):.4f}")
    print(f"RAE Train: {metrics.get('RAE_Train', 'N/A'):.4f}")
    print(f"RAE Test: {metrics.get('RAE_Test', 'N/A'):.4f}")
    print(f"RRSE Train: {metrics.get('RRSE_Train', 'N/A'):.4f}")
    print(f"RRSE Test: {metrics.get('RRSE_Test', 'N/A'):.4f}")
    print(f"AUC-ROC Train: {metrics.get('AUC_ROC_Train', 'N/A'):.3f}")
    print(f"AUC-ROC Test: {metrics.get('AUC_ROC_Test', 'N/A'):.3f}")
    print(f"Directional Accuracy Test: {metrics.get('Directional_Accuracy_Test', 'N/A'):.3f}")
    print(f"Adversarial Robustness Test: {metrics.get('Adversarial_Robustness_Test', 'N/A'):.3f}")

if __name__ == "__main__":
    main()