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
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, GATConv

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
        return super().default(obj)

# GRAPH TRANSFORMER MODEL 

class RedrawProjection:
    def __init__(self, model, redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
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


# DATA LOADING AND PREPROCESSING

class DomainDataLoader:
    """Data loader for domain-specific time series data with graph structure"""
    def __init__(self, data_file, device, seq_len=12, out_len=6, correlation_threshold=0.3, max_edges_per_node=10):
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
        print(f"Loading domain data from: {self.data_file}")

        try:
            df = pd.read_csv(self.data_file, header=0)
            print(f"Loaded {df.shape[1]} columns from header")
        except Exception as e:
            print(f"Error reading with headers: {e}, trying without headers")
            df = pd.read_csv(self.data_file, header=None)
            df.columns = [f"Node_{i}" for i in range(df.shape[1])]
            print(f"Loaded {df.shape[1]} columns without header")

        self.timestamps = df.iloc[:, 0].astype(str).tolist()
        timestamp_col = df.columns[0]

        numeric_df = df.drop(columns=[timestamp_col]).apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )

        # If no valid numeric columns, skip file safely
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric columns found in CSV")

        self.raw_data = numeric_df.to_numpy().astype(float)

        print(f"Numeric data shape: {self.raw_data.shape}")

        scale = np.max(np.abs(self.raw_data), axis=0)
        scale[scale == 0] = 1.0

        self.scale = torch.from_numpy(scale).float()
        self.normalized_data = self.raw_data / self.scale.numpy()

        self.feature_names = numeric_df.columns.tolist()
        self.n, self.m = self.raw_data.shape
        self.col = self.feature_names

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
        """Split data with proper ratios"""
        print("Splitting data with proper ratios: 80% train, 20% validation")
        
        total_samples = len(self.normalized_data)
        n_train = int(total_samples * 0.8)  # 80% for training
        
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {n_train} (80%), Validation samples: {total_samples - n_train} (20%)")
        
        # Split data
        train_data = self.normalized_data[:n_train]
        valid_data = self.normalized_data[n_train:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        
        # Use consistent horizons
        train_out_len = min(self.out_len, len(train_data) - self.seq_len)
        valid_out_len = min(self.out_len, len(valid_data) - self.seq_len)
        
        # Ensure minimum horizons
        train_out_len = max(1, train_out_len)
        valid_out_len = max(1, valid_out_len)
        
        print(f"Forecast horizons - Train: {train_out_len}, Validation: {valid_out_len}")
        
        # Create sequences
        self.train = self._batchify_with_overlap(train_data, self.seq_len, train_out_len, "training")
        self.valid = self._batchify_with_overlap(valid_data, self.seq_len, valid_out_len, "validation")
        
        # Store horizons
        self.train_horizon = train_out_len
        self.valid_horizon = valid_out_len
        
        print(f"Train sequences: {self.train[0].shape[0]} (seq_len: {self.seq_len}, out_len: {train_out_len})")
        print(f"Validation sequences: {self.valid[0].shape[0]} (seq_len: {self.seq_len}, out_len: {valid_out_len})")
        
        # Verify we have sequences
        if self.train[0].shape[0] == 0:
            print("ERROR: No training sequences created!")
        if self.valid[0].shape[0] == 0:
            print("ERROR: No validation sequences created!")
            
    def _batchify_with_overlap(self, data, seq_len, out_len, split_name):
        """Create sequences with maximum overlap"""
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
        
        total_elements = n * out_len * self.m
        print(f"Created {n} {split_name} sequences with {total_elements} total target elements")
        
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    
    def prepare_pyg_data(self, sequences, targets, forecast_horizon=None, walk_length=20):
        """Convert sequences to PyG Data objects"""
        data_list = []
        
        # Use provided forecast_horizon or calculate from targets
        if forecast_horizon is None:
            if len(targets) > 0:
                forecast_horizon = targets[0].shape[0]
            else:
                forecast_horizon = self.out_len
        
        print(f"Preparing PyG data with forecast horizon: {forecast_horizon}")
        print(f"Number of sequences: {len(sequences)}")
        
        for i, (seq, target) in enumerate(zip(sequences, targets)):
            x = torch.FloatTensor(seq[-1])
            
            transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
            
            # Verify target shape
            actual_horizon = target.shape[0]
            if actual_horizon != forecast_horizon:
                print(f"Warning: Sequence {i} - Target horizon {actual_horizon} doesn't match expected {forecast_horizon}")
                current_horizon = actual_horizon
            else:
                current_horizon = forecast_horizon
            
            # Flatten target: (horizon * num_nodes)
            target_flat = torch.FloatTensor(target).flatten()
            
            # Verify flattened target size
            expected_flat_size = current_horizon * self.m
            if target_flat.numel() != expected_flat_size:
                print(f"Error: Sequence {i} - Flattened target size {target_flat.numel()} doesn't match expected {expected_flat_size}")
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

class CyberThreatDataLoader(DomainDataLoader):
    """Extended data loader for cyber threat data with test split"""
    def split_data(self):
        """Split data with proper ratios: 70% train, 15% validation, 15% test"""
        print("Splitting cyber threat data with ratios: 70% train, 15% validation, 15% test")
        
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
        
        # Use consistent horizons
        train_out_len = min(self.out_len, len(train_data) - self.seq_len)
        valid_out_len = min(self.out_len, len(valid_data) - self.seq_len)
        test_out_len = min(self.out_len, len(test_data) - self.seq_len)
        
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
        
        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")

# TRANSFER LEARNING COMPONENTS

class DomainPreTrainer:
    """Handles pre-training on a single domain"""
    
    def __init__(self, domain_name, model_class, base_config, device):
        self.domain_name = domain_name
        self.model_class = model_class
        self.base_config = base_config.copy()
        self.device = device
        self.best_state = None
        self.training_history = []
    
    def load_and_prepare_data(self, data_path, sequence_length=12, forecast_horizon=6):
        """Load domain data and prepare for training"""
        print(f"\nLoading {self.domain_name} data from {data_path}")
        
        try:
            data_loader = DomainDataLoader(
                data_path, 
                self.device, 
                seq_len=sequence_length, 
                out_len=forecast_horizon,
                correlation_threshold=0.3,
                max_edges_per_node=10
            )
            
            print(f"  Data loaded: {data_loader.m} nodes, {len(data_loader.train[0])} training sequences")
            return data_loader
            
        except Exception as e:
            print(f"  ERROR loading {self.domain_name}: {e}")
            return None
    
    def pretrain(self, data_loader, epochs=100, batch_size=8, learning_rate=0.001):
        """Pre-train model on this domain"""
        if data_loader is None:
            print(f"  SKIPPING {self.domain_name} - no valid data")
            return None
            
        print(f"\n{'='*80}")
        print(f"PRE-TRAINING ON {self.domain_name.upper()}")
        print(f"{'='*80}")
        
        # Adjust config for this domain
        config = self.base_config.copy()
        config['num_nodes'] = data_loader.m
        config['forecast_horizon'] = data_loader.train_horizon
        
        print(f"  Model config: nodes={config['num_nodes']}, horizon={config['forecast_horizon']}")
        
        # Create model
        model = self.model_class(config).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Prepare data
        train_data = data_loader.prepare_pyg_data(
            data_loader.train[0].numpy(), 
            data_loader.train[1].numpy(), 
            walk_length=config['pe_walk_length']
        )
        
        if len(train_data) == 0:
            print(f"  No training data available for {self.domain_name}")
            return None
            
        train_loader = PyGDataLoader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True)
        
        # Validation data
        valid_data = data_loader.prepare_pyg_data(
            data_loader.valid[0].numpy(),
            data_loader.valid[1].numpy(),
            walk_length=config['pe_walk_length']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, 
                             mc_dropout=True, forecast_horizon=data_loader.train_horizon)
                
                # Reshape target
                batch_size = output.shape[0]
                target_reshaped = batch.y.view(batch_size, data_loader.train_horizon, data_loader.m)
                
                loss = criterion(output, target_reshaped)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
            
            if batch_count == 0:
                continue
                
            avg_train_loss = train_loss / batch_count
            
            # Validation
            model.eval()
            val_loss = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in PyGDataLoader(valid_data, batch_size=min(batch_size, len(valid_data)), shuffle=False):
                    batch = batch.to(self.device)
                    output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch,
                                 mc_dropout=False, forecast_horizon=data_loader.valid_horizon)
                    
                    batch_size = output.shape[0]
                    target_reshaped = batch.y.view(batch_size, data_loader.valid_horizon, data_loader.m)
                    
                    val_loss += criterion(output, target_reshaped).item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
            scheduler.step(avg_val_loss)
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_state = model.state_dict().copy()
                patience_counter = 0
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f} *")
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
            
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"\nPre-training completed. Best val loss: {best_val_loss:.6f}")
        return self.best_state

class TransferManager:
    """Manages transfer of pre-trained weights to cyber threat model"""
    
    def __init__(self, pretrained_states, target_config, model_class, device):
        self.pretrained_states = pretrained_states
        self.target_config = target_config
        self.model_class = model_class
        self.device = device
    
    def create_transferred_model(self, strategy='selective'):
        """
        Create cyber threat model with transferred weights
        
        Args:
            strategy: 'single' (use best domain), 'ensemble' (average weights), 'selective' (cherry-pick layers)
        """
        print(f"\n{'='*80}")
        print(f"TRANSFERRING TO CYBER THREAT MODEL (Strategy: {strategy})")
        print(f"{'='*80}")
        
        # Create target model
        target_model = self.model_class(self.target_config).to(self.device)
        target_state = target_model.state_dict()
        
        if strategy == 'single':
            return self._transfer_single(target_model, target_state)
        elif strategy == 'ensemble':
            return self._transfer_ensemble(target_model, target_state)
        elif strategy == 'selective':
            return self._transfer_selective(target_model, target_state)
        else:
            return target_model, []
    
    def _transfer_single(self, target_model, target_state):
        """Transfer from single best pre-trained model"""
        if not self.pretrained_states:
            print("No pre-trained models available for transfer")
            return target_model, []
            
        source_state = self.pretrained_states[0]['state_dict']
        transferred = self._transfer_compatible_weights(source_state, target_state)
        target_model.load_state_dict(target_state)
        
        print(f"Transferred {len(transferred)} layers from {self.pretrained_states[0]['domain']}")
        return target_model, transferred
    
    def _transfer_ensemble(self, target_model, target_state):
        """Average weights from multiple pre-trained models"""
        print("Averaging weights from multiple pre-trained models...")
        
        if not self.pretrained_states:
            print("No pre-trained models available for ensemble")
            return target_model, []
        
        # Collect weights to average
        weights_to_average = {}
        
        for pretrained in self.pretrained_states:
            source_state = pretrained['state_dict']
            
            for name, param in source_state.items():
                if name in target_state and target_state[name].shape == param.shape:
                    if name not in weights_to_average:
                        weights_to_average[name] = []
                    weights_to_average[name].append(param.cpu())
        
        # Average and transfer
        transferred = []
        for name, weight_list in weights_to_average.items():
            if len(weight_list) > 0:
                avg_weight = torch.stack(weight_list).mean(dim=0)
                target_state[name] = avg_weight
                transferred.append(name)
        
        target_model.load_state_dict(target_state)
        print(f"Averaged and transferred {len(transferred)} layers from {len(self.pretrained_states)} domains")
        
        return target_model, transferred
    
    def _transfer_selective(self, target_model, target_state):
        """Selectively transfer only transformer layers, not domain-specific layers"""
        print("Selectively transferring transformer backbone...")
        
        if not self.pretrained_states:
            print("No pre-trained models available for selective transfer")
            return target_model, []
            
        # Only transfer transformer encoder layers, not input/output projections
        layers_to_transfer = ['convs', 'norm', 'pe_lin', 'pe_norm']
        layers_to_skip = ['node_emb', 'edge_emb', 'forecast_head']
        
        source_state = self.pretrained_states[0]['state_dict']
        transferred = []
        
        for name, param in source_state.items():
            # Check if this layer should be transferred
            should_transfer = any(layer in name for layer in layers_to_transfer)
            should_skip = any(skip in name for skip in layers_to_skip)
            
            if should_transfer and not should_skip:
                if name in target_state and target_state[name].shape == param.shape:
                    target_state[name] = param
                    transferred.append(name)
        
        target_model.load_state_dict(target_state)
        print(f"Selectively transferred {len(transferred)} transformer layers")
        
        return target_model, transferred
    
    def _transfer_compatible_weights(self, source_state, target_state):
        """Transfer all compatible weights"""
        transferred = []
        
        for name, param in source_state.items():
            if name in target_state:
                if target_state[name].shape == param.shape:
                    target_state[name] = param
                    transferred.append(name)
        
        return transferred

def train_model_finetune(data, model, optimizer, criterion, batch_size, device, epochs, config):
    """Fine-tune the transferred model on cyber threat data"""
    model.train()
    
    train_data = data.prepare_pyg_data(data.train[0].numpy(), data.train[1].numpy(), walk_length=config['pe_walk_length'])
    if len(train_data) == 0:
        print("No training data available for fine-tuning")
        return float('inf')
        
    train_loader = PyGDataLoader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, 
                         mc_dropout=True, forecast_horizon=data.train_horizon)
            
            # Calculate horizon properly
            batch_size_actual = output.shape[0]
            expected_horizon = data.train_horizon
            
            # Verify the target size matches expected
            expected_target_size = batch_size_actual * expected_horizon * data.m
            actual_target_size = batch.y.numel()
            
            if actual_target_size != expected_target_size:
                if batch_size_actual > 0 and data.m > 0:
                    actual_horizon = actual_target_size // (batch_size_actual * data.m)
                    if actual_horizon * batch_size_actual * data.m == actual_target_size:
                        expected_horizon = actual_horizon
                    else:
                        continue
                else:
                    continue
            
            # Reshape target to match output
            target_reshaped = batch.y.view(batch_size_actual, expected_horizon, data.m)
            
            # Ensure shapes match
            if output.shape != target_reshaped.shape:
                min_horizon = min(output.shape[1], target_reshaped.shape[1])
                if min_horizon > 0:
                    output = output[:, :min_horizon, :]
                    target_reshaped = target_reshaped[:, :min_horizon, :]
                else:
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
        
        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Fine-tune Epoch {epoch}: Average Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return best_loss


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Pre-training for Graph Transformer')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=50, help='epochs for pre-training')
    parser.add_argument('--epochs_finetune', type=int, default=20, help='epochs for initial fine-tuning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--transfer_strategy', type=str, default='selective', choices=['single', 'ensemble', 'selective'], help='transfer learning strategy')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs('transfer_learning_pretrain', exist_ok=True)
    
    # Set random seed
    set_random_seed(123)
    
    # Base configuration for pre-training
    base_config = {
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
        'learning_rate': 0.001,
        'batch_size': 8,
        'grad_clip': 1.0,
        'forecast_horizon': 6,  # For pre-training
        'local_gnn_type': 'GINE',
        'norm_type': 'batch',
        'head_hidden_dims': [128, 64],
        'head_activation': 'relu'
    }
    
    #  PRE-TRAINING 
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON RELATED DOMAINS")
    print("="*80)
    
    # Find all CSV files in pretrain directory
    pretrain_files = glob.glob(os.path.join(args.pretrain_data_dir, "*.csv"))
    print(f"Found {len(pretrain_files)} pre-training files")
    
    pretrained_models = []
    
    for file_path in pretrain_files:
        domain_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing domain: {domain_name}")
        
        pretrainer = DomainPreTrainer(
            domain_name, GraphTransformer, base_config, device
        )
        
        data_loader = pretrainer.load_and_prepare_data(file_path, sequence_length=12, forecast_horizon=6)
        pretrained_state = pretrainer.pretrain(
            data_loader, 
            epochs=args.epochs_pretrain, 
            batch_size=base_config['batch_size'],
            learning_rate=base_config['learning_rate']
        )
        
        if pretrained_state is not None:
            pretrained_models.append({
                'domain': domain_name,
                'state_dict': pretrained_state,
                'training_history': pretrainer.training_history,
                'data_info': {
                    'num_nodes': data_loader.m,
                    'num_sequences': len(data_loader.train[0])
                }
            })
            
            # Save pre-trained model
            torch.save({
                'state_dict': pretrained_state,
                'config': base_config,
                'domain': domain_name,
                'training_history': pretrainer.training_history
            }, f'transfer_learning_pretrain/pretrained_{domain_name}.pt')
            
            print(f"Saved pre-trained model: transfer_learning_pretrain/pretrained_{domain_name}.pt")
    
    print(f"\nPre-training completed. {len(pretrained_models)} models trained successfully.")
    
    # Save pre-training summary
    pretrain_summary = {
        'total_domains': len(pretrain_files),
        'successful_domains': len(pretrained_models),
        'domains': [p['domain'] for p in pretrained_models],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('transfer_learning_pretrain/pretraining_summary.json', 'w') as f:
        json.dump(pretrain_summary, f, indent=2, cls=NumpyEncoder)
    
    # TRANSFER + INITIAL FINE-TUNING 
    if pretrained_models:
        print("\n" + "="*80)
        print("PHASE 2: TRANSFER + INITIAL FINE-TUNING")
        print("="*80)
        
        # Load cyber threat data
        print("Loading cyber threat data for fine-tuning...")
        try:
            cyber_data_loader = CyberThreatDataLoader(
                args.cyber_data, 
                device, 
                seq_len=12, 
                out_len=36,
                correlation_threshold=0.3,
                max_edges_per_node=15
            )
            
            print(f"Cyber threat data loaded: {cyber_data_loader.m} nodes")
            print(f"Training sequences: {len(cyber_data_loader.train[0])}")
            print(f"Validation sequences: {len(cyber_data_loader.valid[0])}")
            print(f"Test sequences: {len(cyber_data_loader.test[0])}")
            
        except Exception as e:
            print(f"Error loading cyber threat data: {e}")
            return
        
        # Cyber threat model config 
        cyber_config = base_config.copy()
        cyber_config['num_nodes'] = cyber_data_loader.m
        cyber_config['forecast_horizon'] = cyber_data_loader.train_horizon
        
        print(f"Cyber threat model config: nodes={cyber_config['num_nodes']}, horizon={cyber_config['forecast_horizon']}")
        
        # Transfer weights
        transfer_manager = TransferManager(
            pretrained_models, cyber_config, GraphTransformer, device
        )
        
        transferred_model, transferred_layers = transfer_manager.create_transferred_model(
            strategy=args.transfer_strategy
        )
        
        # Initial fine-tuning
        print(f"\nInitial fine-tuning for {args.epochs_finetune} epochs...")
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(transferred_model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower LR for fine-tuning
        
        best_val_loss = float('inf')
        best_state = None
        
        # Prepare validation data for monitoring
        valid_data = cyber_data_loader.prepare_pyg_data(
            cyber_data_loader.valid[0].numpy(),
            cyber_data_loader.valid[1].numpy(),
            walk_length=cyber_config['pe_walk_length']
        )
        
        for epoch in range(1, args.epochs_finetune + 1):
            # Training
            train_loss = train_model_finetune(
                cyber_data_loader, transferred_model, optimizer, criterion, 
                cyber_config['batch_size'], device, 1, cyber_config
            )
            
            if math.isinf(train_loss):
                continue
            
            # Validation
            transferred_model.eval()
            val_loss = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in PyGDataLoader(valid_data, batch_size=min(cyber_config['batch_size'], len(valid_data)), shuffle=False):
                    batch = batch.to(device)
                    output = transferred_model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch,
                                             mc_dropout=False, forecast_horizon=cyber_data_loader.valid_horizon)
                    
                    batch_size_actual = output.shape[0]
                    target_reshaped = batch.y.view(batch_size_actual, cyber_data_loader.valid_horizon, cyber_data_loader.m)
                    
                    val_loss += criterion(output, target_reshaped).item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
            
            print(f"Fine-tune Epoch {epoch:3d}: Train={train_loss:.6f}, Val={avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = transferred_model.state_dict().copy()
        
        # Save the fine-tuned model
        model_save_data = {
            'model_state_dict': best_state,
            'config': cyber_config,
            'transferred_layers': transferred_layers,
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'transfer_strategy': args.transfer_strategy,
            'fine_tune_val_loss': best_val_loss,
            'cyber_data_info': {
                'num_nodes': cyber_data_loader.m,
                'train_sequences': len(cyber_data_loader.train[0]),
                'valid_sequences': len(cyber_data_loader.valid[0]),
                'test_sequences': len(cyber_data_loader.test[0])
            }
        }
        
        torch.save(model_save_data, 'transfer_learning_pretrain/transferred_model_finetuned.pt')
        
        print(f"\nPhase 2 completed. Fine-tuned model saved.")
        print(f"Final validation loss: {best_val_loss:.6f}")
        print(f"Transferred layers: {len(transferred_layers)}")
        print(f"Model saved as: transfer_learning_pretrain/transferred_model_finetuned.pt")
    
    else:
        print("No pre-trained models available for transfer learning.")
    
    print(f"\nTransfer learning pipeline completed.")
    print(f"Results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()