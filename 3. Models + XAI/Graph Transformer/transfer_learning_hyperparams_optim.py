import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, GATConv
from torch_geometric.nn.attention import PerformerAttention
from sklearn.preprocessing import StandardScaler
import glob

plt.rcParams['savefig.dpi'] = 1200

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class RedrawProjection:
    def __init__(self, model, redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
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
        
        self.pred_len = config['forecast_horizon']
        self.channels = config['channels']
        self.num_nodes = config['num_nodes']
        self.local_gnn_type = config.get('local_gnn_type', 'GINE')
        self.pe_walk_length = config['pe_walk_length']
        
        # Node embedding
        self.node_emb = nn.Linear(config['node_dim'], config['channels'] - config['pe_dim'])
        self.pe_lin = nn.Linear(config['pe_walk_length'], config['pe_dim'])
        
        # Initialize pe_norm properly from the start
        self.pe_norm = nn.BatchNorm1d(config['pe_walk_length'])
        
        # Edge embedding - only for GINE which uses edge attributes
        self.edge_emb = nn.Embedding(2, config['channels'])
        
        # GPS Convolution layers with proper configuration
        self.convs = nn.ModuleList()
        for _ in range(config['num_layers']):
            # Create local GNN for this layer
            local_gnn = self._create_local_gnn(config)
            
            # Create attention kwargs
            attn_kwargs = {'dropout': config['attn_dropout']}
            
            try:
                conv = GPSConv(
                    config['channels'], 
                    local_gnn, 
                    heads=config['num_heads'],
                    attn_type='multihead',
                    attn_kwargs=attn_kwargs
                )
                self.convs.append(conv)
            except Exception as e:
                print(f"GPSConv creation failed: {e}. Using multihead fallback.")
                conv = GPSConv(
                    config['channels'], 
                    local_gnn, 
                    heads=config['num_heads'],
                    attn_type='multihead',
                    attn_kwargs={'dropout': config['attn_dropout']}
                )
                self.convs.append(conv)
        
        # Normalization
        norm_type = config.get('norm_type', 'batch')
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(config['channels'])
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(config['channels'])
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(config['channels'])
        else:
            self.norm = nn.BatchNorm1d(config['channels'])
        
        self.dropout = nn.Dropout(config['dropout'])
        
        # Forecasting head
        head_hidden_dims = config.get('head_hidden_dims', [128, 64])
        head_layers = []
        prev_dim = config['channels']
        
        activation_name = config.get('head_activation', 'ReLU')
        if activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'gelu':
            activation = nn.GELU()
        elif activation_name.lower() == 'silu':
            activation = nn.SiLU()
        else:
            activation = nn.ReLU()
        
        for hidden_dim in head_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation,
                nn.Dropout(config['dropout'])
            ])
            prev_dim = hidden_dim
        
        head_layers.append(nn.Linear(prev_dim, config['forecast_horizon'] * config['num_nodes']))
        self.forecast_head = nn.Sequential(*head_layers)
        
        # No redraw projection since we're using multihead
        self.redraw_projection = RedrawProjection(self, redraw_interval=None)

    def _create_local_gnn(self, config):
        """Create local GNN with proper configuration"""
        local_gnn_type = config.get('local_gnn_type', 'GINE')
        
        if local_gnn_type == 'GINE':
            def create_gine_nn():
                activation_name = config.get('head_activation', 'ReLU')
                if activation_name.lower() == 'relu':
                    activation = nn.ReLU()
                elif activation_name.lower() == 'gelu':
                    activation = nn.GELU()
                elif activation_name.lower() == 'silu':
                    activation = nn.SiLU()
                else:
                    activation = nn.ReLU()
                    
                return nn.Sequential(
                    nn.Linear(config['channels'], config['channels']),
                    activation,
                    nn.Linear(config['channels'], config['channels']),
                )
            return GINEConv(create_gine_nn())
            
        elif local_gnn_type == 'GCN':
            return GCNConv(config['channels'], config['channels'])
            
        elif local_gnn_type == 'GAT':
            num_heads = config['num_heads']
            if config['channels'] % num_heads != 0:
                adjusted_channels = (config['channels'] // num_heads) * num_heads
                if adjusted_channels == 0:
                    adjusted_channels = num_heads
                print(f"Adjusting GAT channels from {config['channels']} to {adjusted_channels}")
                head_dim = adjusted_channels // num_heads
            else:
                head_dim = config['channels'] // num_heads
                
            return GATConv(config['channels'], head_dim, heads=num_heads, concat=True)
        
        else:
            def create_gine_nn():
                return nn.Sequential(
                    nn.Linear(config['channels'], config['channels']),
                    nn.ReLU(),
                    nn.Linear(config['channels'], config['channels']),
                )
            return GINEConv(create_gine_nn())

    def forward(self, x, pe, edge_index, edge_attr, batch, mc_dropout=True, forecast_horizon=None):
        # Node embeddings with positional encoding
        x_pe = self.pe_norm(pe)
        
        node_emb = self.node_emb(x)
        pe_emb = self.pe_lin(x_pe)
        
        # Concatenate along feature dimension
        x = torch.cat((node_emb, pe_emb), dim=1)
        
        # Only use edge attributes for GINE
        if self.local_gnn_type == 'GINE':
            edge_attr_emb = self.edge_emb(edge_attr)
        else:
            edge_attr_emb = None
        
        # Apply GPS convolutions
        for conv in self.convs:
            if self.local_gnn_type == 'GINE':
                x = conv(x, edge_index, batch, edge_attr=edge_attr_emb)
            else:
                x = conv(x, edge_index, batch)
            
            x = self.norm(x)
            if mc_dropout:
                x = self.dropout(x)
        
        # Global pooling and forecasting
        x = global_add_pool(x, batch)
        
        # Use provided forecast_horizon or default
        current_forecast_horizon = forecast_horizon if forecast_horizon else self.pred_len
        
        # Handle different forecast horizons
        if current_forecast_horizon != self.pred_len:
            temp_projection = nn.Linear(x.size(-1), current_forecast_horizon * self.num_nodes).to(x.device)
            nn.init.xavier_uniform_(temp_projection.weight)
            output = temp_projection(x)
        else:
            output = self.forecast_head(x)
        
        output = output.view(-1, current_forecast_horizon, self.num_nodes)
        return output

class TransferLearningManager:
    """Manages loading and initialization of pre-trained models for hyperparameter optimization"""
    
    def __init__(self, pretrained_model_path, device):
        self.pretrained_model_path = pretrained_model_path
        self.device = device

    # Helper: copy overlapping region from src tensor into a new tensor shaped like dst
    @staticmethod
    def _copy_overlap_tensor(src, dst):
        src_t = src.cpu().detach()
        dst_t = dst.cpu().detach().clone()
        if src_t.numel() == 0 or dst_t.numel() == 0:
            return dst_t
        # compute slices per dim
        src_shape = list(src_t.shape)
        dst_shape = list(dst_t.shape)
        slices = []
        for s, d in zip(src_shape, dst_shape):
            m = min(s, d)
            slices.append(slice(0, m))
        # build index tuple
        idx = tuple(slices)
        dst_t[idx] = src_t[idx]
        return dst_t

    # Helper: find best candidate target key for a pretrained key using suffix matching and overlap heuristic
    @staticmethod
    def _find_best_target_key(pre_key, target_state_dict):
        # prefer exact match (should be handled before), else try suffix match
        pre_last = pre_key.split('.')[-1]
        candidates = []
        for tkey in target_state_dict.keys():
            if pre_last == tkey.split('.')[-1]:
                candidates.append(tkey)
        # if no exact last-token matches, try any token overlap
        if not candidates:
            pre_tokens = set(pre_key.split('.'))
            for tkey in target_state_dict.keys():
                t_tokens = set(tkey.split('.'))
                if pre_tokens & t_tokens:
                    candidates.append(tkey)
        # rank candidates by overlap of shapes (bigger overlap wins)
        best = None
        best_score = -1
        for c in candidates:
            try:
                pre_shape = None
                # caller should pass the pre tensor separately; here we only estimate by token match
                # fallback: prefer candidates with similar number of dimensions
                score = 1
                if score > best_score:
                    best_score = score
                    best = c
            except Exception:
                continue
        return best

    def load_pretrained_model(self, target_config):
        """Load pre-trained model and adapt it to target configuration"""
        print(f"Loading pre-trained model from: {self.pretrained_model_path}")
        
        try:
            # Load the pre-trained model
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            pretrained_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Create model with target configuration
            model = GraphTransformer(target_config).to(self.device)
            current_state_dict = model.state_dict()
            
            transferred_layers = []
            partially_transferred = []
            skipped = []
            mapped = {}
            
            # First pass: exact-name matches
            for name, param in pretrained_state_dict.items():
                if name in current_state_dict:
                    target_param = current_state_dict[name]
                    if list(param.shape) == list(target_param.shape):
                        current_state_dict[name] = param
                        transferred_layers.append(name)
                    else:
                        # shapes differ: copy overlapping block
                        try:
                            new_param = self._copy_overlap_tensor(param, target_param)
                            current_state_dict[name] = new_param.to(target_param.device)
                            partially_transferred.append((name, tuple(param.shape), tuple(target_param.shape)))
                            transferred_layers.append(name)
                        except Exception as e:
                            print(f"Failed partial copy for {name}: {e}")
                            skipped.append(name)
                else:
                    # not exact match - defer to second pass mapping
                    continue
            
            # Second pass: try to map pretrained keys to target keys heuristically
            for name, param in pretrained_state_dict.items():
                if name in transferred_layers or name in skipped:
                    continue
                # try to find best candidate in current_state_dict not already filled by pretrained
                candidate = self._find_best_target_key(name, {k:v for k,v in current_state_dict.items() if k not in transferred_layers})
                if candidate:
                    target_param = current_state_dict[candidate]
                    try:
                        new_param = self._copy_overlap_tensor(param, target_param)
                        current_state_dict[candidate] = new_param.to(target_param.device)
                        mapped[name] = candidate
                        transferred_layers.append(name)
                        partially_transferred.append((f"{name} -> {candidate}", tuple(param.shape), tuple(target_param.shape)))
                    except Exception as e:
                        print(f"Failed mapping copy {name} -> {candidate}: {e}")
                        skipped.append(name)
                else:
                    skipped.append(name)
            
            # Load merged state dict into model (non-strict to allow unmatched keys)
            model.load_state_dict(current_state_dict, strict=False)
            
            # Logging summary
            print(f"Total pretrained keys: {len(pretrained_state_dict)}")
            print(f"Fully/partially transferred keys: {len(transferred_layers)}")
            if partially_transferred:
                print("Partially transferred keys (name, pretrained_shape, target_shape):")
                for info in partially_transferred[:50]:
                    print(f"  {info}")
            if mapped:
                print("Mapped keys from pretrained -> target:")
                for p,k in mapped.items():
                    print(f"  {p} -> {k}")
            if skipped:
                print(f"Skipped keys (could not map or copy): {len(skipped)} - sample: {skipped[:10]}")
            
            return model, transferred_layers
            
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            # Fallback: create fresh model
            print("Creating fresh model as fallback")
            return GraphTransformer(target_config).to(self.device), []

class CyberThreatDataLoader:
    def __init__(self, data_file, device, seq_len=12, out_len=36, correlation_threshold=0.3, max_edges_per_node=15):
        self.data_file = data_file
        self.device = device
        self.seq_len = seq_len
        self.out_len = out_len
        self.correlation_threshold = correlation_threshold
        self.max_edges_per_node = max_edges_per_node
        
        self.P = seq_len
        self.h = out_len
        
        self.read_data()
        self.create_graph_structure()
        self.split_data()  
        
    def read_data(self):
        print(f"Loading data from: {self.data_file}")
        try:
            self.data = pd.read_csv(self.data_file, header=0)
            self.feature_names = self.data.columns.tolist()
            print(f"Loaded {len(self.feature_names)} nodes from header")
        except Exception as e:
            print(f"Error reading with headers: {e}, trying without headers")
            self.data = pd.read_csv(self.data_file, header=None)
            self.feature_names = [f"Node_{i}" for i in range(self.data.shape[1])]
            
        self.raw_data = self.data.values
        print(f"Raw data shape: {self.raw_data.shape}")

        self.n, self.m = self.raw_data.shape
        self.col = self.feature_names

        # Create date range from July 2011 to December 2024
        self.dates = []
        current_date = datetime(2011, 7, 1)
        end_date = datetime(2024, 12, 1)
        while current_date <= end_date:
            self.dates.append(current_date)
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        print(f"Date range: {self.dates[0].strftime('%Y-%m')} to {self.dates[-1].strftime('%Y-%m')}")
        print(f"Total months: {len(self.dates)}")

        # Normalize data
        self.scale = torch.from_numpy(np.max(np.abs(self.raw_data), axis=0)).float()
        self.scale[self.scale == 0] = 1.0
            
        self.normalized_data = self.raw_data / self.scale.numpy()
        print(f"Normalized data shape: {self.normalized_data.shape}")
        print(f"Time steps: {self.n}, Nodes: {self.m}")
        
    def create_graph_structure(self):
        """Create graph structure based on correlation between nodes"""
        print(f"Creating graph structure with threshold={self.correlation_threshold}, max_edges={self.max_edges_per_node}")
        
        max_samples_for_corr = min(100, self.n)
        data_for_corr = self.normalized_data[:max_samples_for_corr]
        data_for_corr = data_for_corr + 1e-8
        
        try:
            correlation_matrix = np.corrcoef(data_for_corr, rowvar=False)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        except:
            print("Correlation calculation failed, using identity matrix")
            correlation_matrix = np.eye(self.m)
        
        adj_matrix = (np.abs(correlation_matrix) > self.correlation_threshold).astype(int)
        
        edge_counts = np.zeros(self.m)
        edge_index = []
        edge_attr = []
        
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if (adj_matrix[i, j] == 1 and 
                    edge_counts[i] < self.max_edges_per_node and 
                    edge_counts[j] < self.max_edges_per_node):
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(1)
                    edge_attr.append(1)
                    edge_counts[i] += 1
                    edge_counts[j] += 1
        
        if not edge_index:
            print("Warning: No edges found with current threshold. Creating a sparse connected graph.")
            for i in range(min(50, self.m)):
                for j in range(i + 1, min(i + 6, self.m)):
                    if edge_counts[i] < self.max_edges_per_node and edge_counts[j] < self.max_edges_per_node:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        edge_attr.append(1)
                        edge_attr.append(1)
                        edge_counts[i] += 1
                        edge_counts[j] += 1
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        print(f"Graph created: {self.edge_index.shape[1]} edges")
        print(f"Average edges per node: {self.edge_index.shape[1] / self.m:.2f}")
        
    def split_data(self):
        """Split data with proper ratios and ensure minimum sequences and consistent horizons"""
        print("Splitting data with proper ratios: 70% train, 15% validation, 15% test")
        
        total_samples = len(self.normalized_data)
        n_train = int(total_samples * 0.7)  # 70% for training
        n_valid = int(total_samples * 0.15) # 15% for validation
        
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {n_train} (70%), Validation samples: {n_valid} (15%)")
        print(f"Test samples: {total_samples - n_train - n_valid} (15%)")
        
        # Split data
        train_data = self.normalized_data[:n_train]
        valid_data = self.normalized_data[n_train:n_train + n_valid]
        test_data = self.normalized_data[n_train + n_valid:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Use consistent horizons that work with the data length
        train_out_len = min(36, len(train_data) - self.seq_len)  # Max 36 months for training
        valid_out_len = min(12, len(valid_data) - self.seq_len)  # Max 12 months for validation
        test_out_len = min(12, len(test_data) - self.seq_len)    # Max 12 months for testing
        
        # Ensure minimum horizons
        train_out_len = max(1, train_out_len)
        valid_out_len = max(1, valid_out_len)
        test_out_len = max(1, test_out_len)
        
        print(f"Forecast horizons - Train: {train_out_len}, Validation: {valid_out_len}, Test: {test_out_len}")
        
        # Create sequences
        self.train = self._batchify_with_overlap(train_data, self.seq_len, train_out_len, "training")
        self.valid = self._batchify_with_overlap(valid_data, self.seq_len, valid_out_len, "validation")
        self.test = self._batchify_with_overlap(test_data, self.seq_len, test_out_len, "testing")
        
        # Store horizons
        self.train_horizon = train_out_len
        self.valid_horizon = valid_out_len
        self.test_horizon = test_out_len
        
        print(f"Train sequences: {self.train[0].shape[0]} (seq_len: {self.seq_len}, out_len: {train_out_len})")
        print(f"Validation sequences: {self.valid[0].shape[0]} (seq_len: {self.seq_len}, out_len: {valid_out_len})")
        print(f"Test sequences: {self.test[0].shape[0]} (seq_len: {self.seq_len}, out_len: {test_out_len})")
        
        # Verify we have sequences
        if self.train[0].shape[0] == 0:
            print("ERROR: No training sequences created!")
        if self.valid[0].shape[0] == 0:
            print("ERROR: No validation sequences created!")
        if self.test[0].shape[0] == 0:
            print("ERROR: No test sequences created!")
            
    def _batchify_with_overlap(self, data, seq_len, out_len, split_name):
        """Create sequences with maximum overlap and validation"""
        available_samples = len(data)
        required_samples = seq_len + out_len
        
        print(f"{split_name.capitalize()} - Available: {available_samples}, Required: {required_samples}")
        
        if available_samples < required_samples:
            print(f"Warning: Not enough {split_name} data for full sequences.")
            if available_samples >= out_len + 6:
                actual_seq_len = available_samples - out_len
                n = 1
                X = np.zeros((n, actual_seq_len, self.m))
                Y = np.zeros((n, out_len, self.m))
                X[0] = data[:actual_seq_len]
                Y[0] = data[actual_seq_len:actual_seq_len + out_len]
                print(f"Created {split_name} sequence with input length {actual_seq_len} and output length {out_len}")
                return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
            else:
                print(f"Cannot create {split_name} sequences with available data")
                return torch.zeros(0, seq_len, self.m), torch.zeros(0, out_len, self.m)
        
        n = len(data) - seq_len - out_len + 1
        if n <= 0:
            print(f"Warning: Not enough {split_name} data to create sequences.")
            return torch.zeros(0, seq_len, self.m), torch.zeros(0, out_len, self.m)
            
        X = np.zeros((n, seq_len, self.m))
        Y = np.zeros((n, out_len, self.m))
        
        for i in range(n):
            X[i] = data[i:i+seq_len]
            Y[i] = data[i+seq_len:i+seq_len+out_len]
        
        # Verify the created sequences
        total_elements = n * out_len * self.m
        print(f"Created {n} {split_name} sequences with {total_elements} total target elements")
        
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    
    def prepare_pyg_data(self, sequences, targets, forecast_horizon=None, walk_length=20):
        """Convert sequences to PyG Data objects with proper target handling"""
        data_list = []
        
        # Use provided forecast_horizon or calculate from targets
        if forecast_horizon is None:
            # Calculate from target shape
            if len(targets) > 0:
                forecast_horizon = targets[0].shape[0]
            else:
                forecast_horizon = self.out_len
        
        print(f"Preparing PyG data with forecast horizon: {forecast_horizon}")
        print(f"Number of sequences: {len(sequences)}, Target shape: {targets[0].shape if len(targets) > 0 else 'None'}")
        
        for i, (seq, target) in enumerate(zip(sequences, targets)):
            x = torch.FloatTensor(seq[-1])
            
            transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
            
            # Verify target shape
            actual_horizon = target.shape[0]
            if actual_horizon != forecast_horizon:
                print(f"Warning: Sequence {i} - Target horizon {actual_horizon} doesn't match expected {forecast_horizon}")
                # Use the actual target horizon for this sequence
                current_horizon = actual_horizon
            else:
                current_horizon = forecast_horizon
            
            # Flatten target: (horizon * num_nodes)
            target_flat = torch.FloatTensor(target).flatten()
            
            # Verify flattened target size
            expected_flat_size = current_horizon * self.m
            if target_flat.numel() != expected_flat_size:
                print(f"Error: Sequence {i} - Flattened target size {target_flat.numel()} doesn't match expected {expected_flat_size}")
                print(f"Target shape: {target.shape}, Horizon: {current_horizon}, Nodes: {self.m}")
                continue
            
            data = Data(
                x=x.unsqueeze(1),
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=target_flat,
                num_nodes=self.m,
                forecast_horizon=current_horizon
            )
            data = transform(data)
            data_list.append(data)
        
        print(f"Successfully prepared {len(data_list)} PyG data objects")
        return data_list

def create_directories():
    directories = [
        'model/GraphTransformer/Validation',
        'model/GraphTransformer/Testing', 
        'model/GraphTransformer/plots',
        'model/GraphTransformer/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def consistent_name(name):
    if name in ['CAPTCHA', 'DNSSEC', 'RRAM']:
        return name
        
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            if i < len(words) - 1:
                result += ' '
        return result
        
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word in ['MITM', 'SIEM']:
            result += word
        else:
            result += word[0] + word[1:].lower()
        if i < len(words) - 1:
            result += ' '
    return result

def save_metrics_1d(predict, test, title, type):
    if predict.numel() == 0 or test.numel() == 0:
        return
        
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff.item())

    sum_absolute_diff = torch.sum(torch.abs(test - predict))

    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r.item())

    rrse = root_sum_squared / root_sum_squared_r if root_sum_squared_r > 0 else float('inf')

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = sum_absolute_diff / sum_absolute_r if sum_absolute_r > 0 else float('inf')
    rae = rae.item()

    title = title.replace('/', '_')
    with open(f'model/GraphTransformer/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')

def plot_predicted_actual(predicted, actual, title, type, confidence_95=None, dates=None):
    plt.figure(figsize=(15, 6))
    
    if dates is not None and len(dates) == len(predicted):
        x = dates
        # Format dates for x-axis
        date_labels = [d.strftime('%b %Y') for d in dates]
        # Show every 6th label for readability
        display_indices = list(range(0, len(date_labels), max(1, len(date_labels)//12)))
        display_labels = [date_labels[i] for i in display_indices]
    else:
        x = range(1, len(predicted) + 1)
        date_labels = None
    
    plt.plot(x, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted, '--', color='purple', label='Predicted', linewidth=2)
    
    if confidence_95 is not None and len(confidence_95) == len(predicted):
        plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, 
                        alpha=0.3, color='pink', label='95% Confidence')
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(f'{title} - {type}', y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Time", fontsize=15)
    
    if date_labels is not None:
        plt.xticks([x[i] for i in display_indices], display_labels, rotation=45, fontsize=10)
    else:
        plt.xticks(fontsize=13)
    
    plt.yticks(fontsize=13)
    
    title_clean = title.replace('/', '_')
    plt.savefig(f'model/GraphTransformer/{type}/{title_clean}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/GraphTransformer/{type}/{title_clean}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()

def calculate_rrse_rae_comprehensive(predict, test, scale):
    if predict.numel() == 0 or test.numel() == 0:
        return float('inf'), float('inf')
    
    try:
        predict_denorm = predict * scale
        test_denorm = test * scale
        
        predict_flat = predict_denorm.flatten()
        test_flat = test_denorm.flatten()
        
        squared_errors = (test_flat - predict_flat) ** 2
        sum_squared_errors = torch.sum(squared_errors)
        root_sum_squared_errors = torch.sqrt(sum_squared_errors)
        
        test_mean = torch.mean(test_flat)
        squared_deviations = (test_flat - test_mean) ** 2
        sum_squared_deviations = torch.sum(squared_deviations)
        root_sum_squared_deviations = torch.sqrt(sum_squared_deviations)
        
        if root_sum_squared_deviations > 0:
            rrse = root_sum_squared_errors / root_sum_squared_deviations
        else:
            rrse = float('inf')
        
        absolute_errors = torch.abs(test_flat - predict_flat)
        sum_absolute_errors = torch.sum(absolute_errors)
        
        absolute_deviations = torch.abs(test_flat - test_mean)
        sum_absolute_deviations = torch.sum(absolute_deviations)
        
        if sum_absolute_deviations > 0:
            rae = sum_absolute_errors / sum_absolute_deviations
        else:
            rae = float('inf')
        
        rrse = rrse.item() if isinstance(rrse, torch.Tensor) else rrse
        rae = rae.item() if isinstance(rae, torch.Tensor) else rae
        
        return rrse, rae
        
    except Exception as e:
        print(f"Error in comprehensive metric calculation: {e}")
        return float('inf'), float('inf')

def plot_evaluation_results(predictions, targets, confidence, data, eval_type, plot_dates=None):
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    confidence_np = confidence.numpy()
    
    num_nodes_to_plot = min(645, data.m)
    print(f"Creating {num_nodes_to_plot} {eval_type.lower()} plots with confidence intervals...")

    for col in range(num_nodes_to_plot):
        if col < len(data.col):
            node_name = data.col[col]
        else:
            node_name = f"Node_{col}"
        node_name = consistent_name(node_name)

        if predictions_np.shape[0] > 0:
            # Use the last sequence for plotting (most recent)
            pred_curve = predictions_np[-1, :, col]
            target_curve = targets_np[-1, :, col]
            confidence_curve = confidence_np[-1, :, col]

            save_metrics_1d(torch.from_numpy(pred_curve), torch.from_numpy(target_curve), node_name, eval_type)
            plot_predicted_actual(pred_curve, target_curve, node_name, eval_type, confidence_curve, plot_dates)

def get_lr_scheduler(optimizer, scheduler_type, warmup_epochs, total_epochs):
    """Get learning rate scheduler based on configuration"""
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        scheduler = None
    
    return scheduler

def train_model_with_pretrained(data, model, optimizer, criterion, batch_size, device, epochs, config, transferred_layers):
    """Train model with pre-trained initialization, using different learning rates for different parts"""
    model.train()
    
    train_data = data.prepare_pyg_data(data.train[0].numpy(), data.train[1].numpy(), walk_length=config['pe_walk_length'])
    if len(train_data) == 0:
        print("No training data available")
        return float('inf')
        
    train_loader = PyGDataLoader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True)
    
    # Get learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config.get('lr_scheduler', 'none'), 
                               config.get('warmup_epochs', 0), epochs)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Learning rate warmup
        if epoch < config.get('warmup_epochs', 0):
            lr_scale = min(1.0, float(epoch + 1) / config['warmup_epochs'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group.get('initial_lr', config['learning_rate']) * lr_scale
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            model.redraw_projection.redraw_projections()
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, 
                         mc_dropout=True, forecast_horizon=data.train_horizon)
            
            # Calculate horizon properly - use the training horizon from data
            batch_size = output.shape[0]
            expected_horizon = data.train_horizon  # Use the stored training horizon
            
            # Verify the target size matches expected
            expected_target_size = batch_size * expected_horizon * data.m
            actual_target_size = batch.y.numel()
            
            if actual_target_size != expected_target_size:
                print(f"Warning: Target size mismatch in training. Expected {expected_target_size}, got {actual_target_size}")
                # Calculate actual horizon from target size
                if batch_size > 0 and data.m > 0:
                    actual_horizon = actual_target_size // (batch_size * data.m)
                    if actual_horizon * batch_size * data.m == actual_target_size:
                        expected_horizon = actual_horizon
                        print(f"Using adjusted horizon: {expected_horizon}")
                    else:
                        print(f"Cannot resolve target shape mismatch, skipping batch")
                        continue
                else:
                    print(f"Invalid batch size or node count, skipping batch")
                    continue
            
            # Reshape target to match output
            target_reshaped = batch.y.view(batch_size, expected_horizon, data.m)
            
            # Debug info on first batch
            if epoch == 0 and batch_count == 0:
                print(f"Training - Output shape: {output.shape}, Target shape: {target_reshaped.shape}")
                print(f"Batch size: {batch_size}, Horizon: {expected_horizon}, Num nodes: {data.m}")
                print(f"Batch target elements: {batch.y.numel()}, Expected: {expected_target_size}")
            
            # Ensure shapes match
            if output.shape != target_reshaped.shape:
                print(f"Shape mismatch: output {output.shape} vs target {target_reshaped.shape}")
                # Try to adjust by using the minimum horizon
                min_horizon = min(output.shape[1], target_reshaped.shape[1])
                if min_horizon > 0:
                    output = output[:, :min_horizon, :]
                    target_reshaped = target_reshaped[:, :min_horizon, :]
                    print(f"Adjusted to common horizon: {min_horizon}")
                else:
                    print("Cannot adjust shapes, skipping batch")
                    continue
            
            # Calculate loss
            loss = criterion(output, target_reshaped)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count == 0:
            continue
            
        avg_loss = total_loss / batch_count
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Average Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return best_loss

def evaluate_comprehensive(data, model, eval_type, batch_size, device, is_plot=False, mc_samples=20):
    model.train()  # Keep dropout active for Bayesian estimation
    
    if eval_type == "Validation":
        X, Y = data.valid
        # Use the actual validation horizon from data
        forecast_horizon = getattr(data, 'valid_horizon', 12)
        if hasattr(data, 'valid_dates') and len(data.valid_dates) >= Y.shape[1]:
            plot_dates = data.valid_dates[-Y.shape[1]:]
        else:
            plot_dates = None
    else:
        X, Y = data.test
        # Use the actual test horizon from data
        forecast_horizon = getattr(data, 'test_horizon', 12)
        if hasattr(data, 'test_dates') and len(data.test_dates) >= Y.shape[1]:
            plot_dates = data.test_dates[-Y.shape[1]:]
        else:
            plot_dates = None
    
    if X.size(0) == 0:
        print(f"No {eval_type} data available for evaluation")
        return float('inf'), float('inf')

    # Use the appropriate forecast horizon for validation/testing
    pyg_data = data.prepare_pyg_data(X.numpy(), Y.numpy(), forecast_horizon)
    
    if len(pyg_data) == 0:
        print(f"No {eval_type} sequences available")
        return float('inf'), float('inf')

    data_loader = PyGDataLoader(pyg_data, batch_size=min(batch_size, len(pyg_data)), shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_variances = []

    batch_count = 0
    for batch in data_loader:
        batch = batch.to(device)
        
        num_runs = mc_samples
        outputs = []

        with torch.no_grad():
            for run in range(num_runs):
                # Use the actual forecast horizon from the data
                current_forecast_horizon = forecast_horizon
                
                output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, 
                             mc_dropout=True, forecast_horizon=current_forecast_horizon)
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean_prediction = torch.mean(outputs, dim=0)
        variance = torch.var(outputs, dim=0)
        std_dev = torch.sqrt(variance)
        confidence_95 = 1.96 * std_dev / math.sqrt(num_runs)

        # Denormalize predictions and targets
        scale_batch = data.scale.expand(mean_prediction.size(0), mean_prediction.size(1), data.m).to(device)
        
        predictions_denorm = mean_prediction * scale_batch
        
        # Use the actual current forecast horizon for reshaping
        current_forecast_horizon = mean_prediction.size(1)  # This is the actual output horizon
        
        # Calculate the expected target size
        expected_target_size = mean_prediction.size(0) * current_forecast_horizon * data.m
        
        # Check if the target size matches
        if batch.y.numel() != expected_target_size:
            print(f"Warning: Target size mismatch. Expected {expected_target_size}, got {batch.y.numel()}")
            print(f"Batch size: {mean_prediction.size(0)}, Horizon: {current_forecast_horizon}, Nodes: {data.m}")
            # Adjust by using the actual target size to calculate the correct horizon
            actual_horizon = batch.y.numel() // (mean_prediction.size(0) * data.m)
            if actual_horizon * mean_prediction.size(0) * data.m == batch.y.numel():
                current_forecast_horizon = actual_horizon
                print(f"Using adjusted horizon: {current_forecast_horizon}")
            else:
                print(f"Cannot resolve target shape mismatch, skipping batch")
                continue
        
        # Reshape targets using the correct horizon
        targets_reshaped = batch.y.view(mean_prediction.size(0), current_forecast_horizon, data.m) * scale_batch
        confidence_denorm = confidence_95 * scale_batch

        all_predictions.append(predictions_denorm.cpu())
        all_targets.append(targets_reshaped.cpu())
        all_variances.append(confidence_denorm.cpu())
        batch_count += 1

    if not all_predictions:
        return float('inf'), float('inf')

    try:
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        all_confidence_tensor = torch.cat(all_variances, dim=0)
        
        rrse, rae = calculate_rrse_rae_comprehensive(all_predictions_tensor, all_targets_tensor, data.scale)
        
        print(f"{eval_type} - RRSE: {rrse:.6f}, RAE: {rae:.6f} (using {current_forecast_horizon}-month horizon)")
        
        if is_plot and all_predictions_tensor.numel() > 0:
            plot_evaluation_results(all_predictions_tensor, all_targets_tensor, all_confidence_tensor, data, eval_type, plot_dates)
            
        return rrse, rae
        
    except Exception as e:
        print(f"Error in {eval_type.lower()} evaluation: {e}")
        return float('inf'), float('inf')

def create_optimizer_with_differential_lr(model, config, transferred_layers):
    """Create optimizer with different learning rates for transferred vs new layers"""
    base_lr = config['learning_rate']
    
    # Parameters for transferred layers (lower learning rate)
    transferred_params = []
    # Parameters for new layers (higher learning rate)
    new_params = []
    
    for name, param in model.named_parameters():
        if any(transferred_layer in name for transferred_layer in transferred_layers):
            transferred_params.append(param)
        else:
            new_params.append(param)
    
    print(f"Transferred parameters: {len(transferred_params)}, New parameters: {len(new_params)}")
    
    # Use lower learning rate for transferred layers, higher for new layers
    optimizer = optim.AdamW([
        {'params': transferred_params, 'lr': base_lr * 0.1},  # Lower LR for fine-tuning
        {'params': new_params, 'lr': base_lr}  # Normal LR for new layers
    ], weight_decay=config['weight_decay'])
    
    # Store initial LR for warmup
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Graph Transformer Hyperparameter Optimization with Pre-trained Model')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--pretrained_model', type=str, default='transfer_learning_pretrain/transferred_model_finetuned.pt', help='path to pre-trained model')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs per iteration')
    parser.add_argument('--iterations', type=int, default=10, help='number of random search iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    create_directories()
    set_random_seed(123)
    
    # Initialize transfer learning manager
    transfer_manager = TransferLearningManager(args.pretrained_model, device)
    
    # Hyperparameter search space 
    search_space = {
        # Architecture Dimensions
        'channels': [32, 64, 96, 128],
        'num_layers': [2, 3, 4],
        
        # Positional Encoding - use consistent walk length
        'pe_dim': [4, 8, 12],
        'pe_walk_length': [20],  # avoid dimension mismatches
        
        # Input Features
        'node_dim': [1],
        
        # Attention Configuration
        'num_heads': [4, 6, 8],
        'attn_type': ['multihead'],  # Only multihead for stability
        'attn_dropout': [0.1, 0.15, 0.2],
        
        # Regularization
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [1e-5, 5e-5, 1e-4],
        
        # Optimization
        'learning_rate': [5e-5, 1e-4, 2.5e-4, 5e-4],
        'lr_scheduler': ['cosine', 'step', 'none'],
        'warmup_epochs': [5, 10],
        
        # Training Configuration
        'batch_size': [4, 8, 12],
        'grad_clip': [0.5, 1.0, 2.0],
        
        # Temporal Configuration
        'forecast_horizon': [36],
        'sequence_length': [12, 15, 18],
        
        # Graph Structure
        'correlation_threshold': [0.25, 0.3, 0.35],
        'max_edges_per_node': [10, 15],
        
        # Monte Carlo Uncertainty
        'mc_samples': [10, 15, 20],
        
        # GPS-specific Configuration
        'local_gnn_type': ['GINE', 'GCN'],  # Remove GAT for stability
        'norm_type': ['batch', 'layer'],
        
        # Forecasting Head Architecture
        'head_hidden_dims': [
            [128, 64],
            [256, 128],
        ],
        'head_activation': ['relu', 'gelu'],
    }
    
    print("Loading data...")
    try:
        data = CyberThreatDataLoader(args.data, device, seq_len=18, out_len=36)
        search_space['num_nodes'] = [data.m]
        
        print(f"Data loaded successfully!")
        print(f"Training sequences: {data.train[0].shape[0]}")
        print(f"Validation sequences: {data.valid[0].shape[0]}")
        print(f"Test sequences: {data.test[0].shape[0]}")
        print(f"Number of nodes: {data.m}")
        
        if data.valid[0].shape[0] == 0:
            print("CRITICAL ERROR: No validation sequences available!")
            return
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize optimization tracking
    optimization_log = {
        'start_time': datetime.now().isoformat(),
        'data_info': {
            'file': args.data,
            'shape': data.raw_data.shape,
            'nodes': data.m,
            'time_steps': data.n,
            'sequence_length': data.seq_len,
            'forecast_horizon': data.out_len,
            'feature_names': data.col[:10] + ['...'] if len(data.col) > 10 else data.col
        },
        'pretrained_model': args.pretrained_model,
        'search_space': search_space,
        'iterations': [],
        'best_results': None,
        'end_time': None,
        'total_successful': 0,
        'total_failed': 0
    }
    
    best_val_rrse = float('inf')
    best_val_rae = float('inf')
    best_hp = None
    best_model_state = None
    best_transferred_layers = []
    
    # Main optimization loop
    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}")
        
        hp = {}
        for key, values in search_space.items():
            hp[key] = random.choice(values)
        
        # Apply constraints to ensure stable configurations
        if hp['local_gnn_type'] == 'GCN' or hp['local_gnn_type'] == 'GAT':
            # Ensure we don't use edge attributes with GCN/GAT
            pass
        
        # Ensure channel divisibility for any potential GAT usage
        if hp['channels'] % hp['num_heads'] != 0:
            # Find compatible channel size
            compatible_channels = [c for c in search_space['channels'] if c % hp['num_heads'] == 0]
            if compatible_channels:
                hp['channels'] = random.choice(compatible_channels)
            else:
                # Adjust num_heads to be compatible
                compatible_heads = [h for h in search_space['num_heads'] if hp['channels'] % h == 0]
                if compatible_heads:
                    hp['num_heads'] = random.choice(compatible_heads)
        
        # Use multihead attention for stability
        hp['attn_type'] = 'multihead'
        
        print(f"Hyperparameters: {hp}")

        iteration_log = {
            'iteration': iteration + 1,
            'hyperparameters': hp,
            'start_time': datetime.now().isoformat(),
            'status': 'failed',
            'error': None,
            'training_loss': None,
            'validation_metrics': None,
            'end_time': None
        }
        
        try:
            # Reinitialize data loader with new parameters if needed
            if 'correlation_threshold' in hp or 'max_edges_per_node' in hp:
                data = CyberThreatDataLoader(
                    args.data, device, 
                    seq_len=hp.get('sequence_length', 18), 
                    out_len=36,
                    correlation_threshold=hp.get('correlation_threshold', 0.3),
                    max_edges_per_node=hp.get('max_edges_per_node', 15)
                )
            
            # Load pre-trained model with current hyperparameters
            model, transferred_layers = transfer_manager.load_pretrained_model(hp)
            criterion = nn.MSELoss()
            
            # Create optimizer with differential learning rates
            optimizer = create_optimizer_with_differential_lr(model, hp, transferred_layers)
            
            train_loss = train_model_with_pretrained(data, model, optimizer, criterion, hp['batch_size'], device, args.epochs, hp, transferred_layers)
            iteration_log['training_loss'] = train_loss
            iteration_log['transferred_layers'] = transferred_layers
            
            if math.isinf(train_loss):
                iteration_log['error'] = 'Training failed - infinite loss'
                print("Training failed, skipping iteration")
                optimization_log['iterations'].append(iteration_log)
                optimization_log['total_failed'] += 1
                continue
                
            val_rrse, val_rae = evaluate_comprehensive(
                data, model, "Validation", hp['batch_size'], device, False, 
                mc_samples=hp.get('mc_samples', 20)
            )
            iteration_log['validation_metrics'] = {'rrse': val_rrse, 'rae': val_rae}
            
            if (not math.isinf(val_rrse) and not math.isnan(val_rrse) and 
                val_rrse < best_val_rrse):
                best_val_rrse = val_rrse
                best_val_rae = val_rae
                best_hp = hp.copy()
                best_model_state = model.state_dict().copy()
                best_transferred_layers = transferred_layers
                iteration_log['status'] = 'success - new best'
                optimization_log['total_successful'] += 1
                print(f"*** NEW GLOBAL BEST! Iteration {iteration + 1} - RRSE: {best_val_rrse:.6f}, RAE: {best_val_rae:.6f} ***")
            else:
                iteration_log['status'] = 'success'
                optimization_log['total_successful'] += 1
                print(f"Iteration {iteration + 1} completed - Val RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
                
        except Exception as e:
            error_msg = str(e)
            iteration_log['error'] = error_msg
            iteration_log['status'] = 'failed'
            optimization_log['total_failed'] += 1
            print(f"Iteration {iteration + 1} failed: {e}")
            import traceback
            traceback.print_exc()
        
        iteration_log['end_time'] = datetime.now().isoformat()
        optimization_log['iterations'].append(iteration_log)
    
    print(" FINAL EVALUATION AND PLOTTING OF BEST MODEL ")
    
    # Save detailed optimization log
    optimization_log['end_time'] = datetime.now().isoformat()
    optimization_log['duration_seconds'] = (datetime.fromisoformat(optimization_log['end_time']) - 
                                          datetime.fromisoformat(optimization_log['start_time'])).total_seconds()
    
    if best_hp is not None and best_model_state is not None:
        optimization_log['best_results'] = {
            'hyperparameters': best_hp,
            'validation_rrse': best_val_rrse,
            'validation_rae': best_val_rae,
            'transferred_layers': best_transferred_layers,
            'model_saved': True
        }
        
        with open('model/GraphTransformer/hp.txt', 'w') as f:
            json.dump(best_hp, f, indent=2, cls=NumpyEncoder)
        
        # Create model with best hyperparameters
        model = GraphTransformer(best_hp).to(device)
        
        # Load state dict with strict=False to handle potential mismatches
        try:
            model.load_state_dict(best_model_state, strict=True)
            print("State dict loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Trying with strict=False...")
            model.load_state_dict(best_model_state, strict=False)
            print("State dict loaded successfully with strict=False")
        
        print("Final validation evaluation with confidence intervals...")
        val_rrse, val_rae = evaluate_comprehensive(
            data, model, "Validation", best_hp['batch_size'], device, True, 
            mc_samples=best_hp.get('mc_samples', 20)
        )
        
        if data.test[0].size(0) > 0:
            print("Final test evaluation with confidence intervals...")
            test_rrse, test_rae = evaluate_comprehensive(
                data, model, "Testing", best_hp['batch_size'], device, True,
                mc_samples=best_hp.get('mc_samples', 20)
            )
        else:
            test_rrse, test_rae = float('inf'), float('inf')
        
        model_save_data = {
            'model_state_dict': model.state_dict(),  # Save the current model state
            'hyperparameters': best_hp,
            'validation_rrse': val_rrse,
            'validation_rae': val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'transferred_layers': best_transferred_layers,
            'pretrained_model': args.pretrained_model,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_save_data, 'model/GraphTransformer/best_hyperparameter_model_pretrained.pt')
        
        print(f"\nOPTIMIZATION COMPLETE")
        print(f"Best Validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if data.test[0].size(0) > 0:
            print(f"Best Testing - RRSE: {test_rrse:.6f}, RAE: {test_rae:.6f}")
        print(f"Transferred layers: {len(best_transferred_layers)}")
        
    else:
        optimization_log['best_results'] = {
            'model_saved': False,
            'reason': 'No successful iterations'
        }
        print("No successful iterations. Creating fallback model...")
        default_hp = {
            'channels': 64,
            'num_layers': 3,
            'pe_dim': 8,
            'pe_walk_length': 20,
            'node_dim': 1,
            'num_heads': 4,
            'attn_type': 'multihead',
            'attn_dropout': 0.2,
            'dropout': 0.3,
            'weight_decay': 1e-4,
            'learning_rate': 0.0005,
            'lr_scheduler': 'cosine',
            'warmup_epochs': 10,
            'batch_size': 8,
            'grad_clip': 1.0,
            'forecast_horizon': 36,
            'sequence_length': 18,
            'correlation_threshold': 0.3,
            'max_edges_per_node': 10,
            'mc_samples': 20,
            'performer_redraw_interval': 1000,
            'performer_nb_features': 128,
            'local_gnn_type': 'GINE',
            'global_model_type': 'Transformer',
            'norm_type': 'batch',
            'head_hidden_dims': [128, 64],
            'head_activation': 'relu',
            'num_nodes': data.m
        }
        
        with open('model/GraphTransformer/hp.txt', 'w') as f:
            json.dump(default_hp, f, indent=2, cls=NumpyEncoder)
        
        model = GraphTransformer(default_hp).to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': default_hp,
            'is_fallback': True,
            'timestamp': datetime.now().isoformat()
        }, 'model/GraphTransformer/best_hyperparameter_model_pretrained.pt')
        
        print("Fallback model saved")

    # Save optimization log
    with open('model/GraphTransformer/hyperparameter_optimization_log_pretrained.json', 'w') as f:
        json.dump(optimization_log, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nHyperparameter optimization with pre-trained model completed.")
    print(f"Successful iterations: {optimization_log['total_successful']}")
    print(f"Failed iterations: {optimization_log['total_failed']}")
    print(f"Detailed log saved: model/GraphTransformer/hyperparameter_optimization_log_pretrained.json")
    print(f"Plots saved in: model/GraphTransformer/Validation/ and model/GraphTransformer/Testing/")
    print(f"Model saved as: model/GraphTransformer/best_hyperparameter_model_pretrained.pt")
    print(f"Hyperparameters saved as: model/GraphTransformer/hp.txt")
    
if __name__ == "__main__":
    main()