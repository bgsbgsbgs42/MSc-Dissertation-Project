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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv
import torch_geometric.transforms as T
import pandas as pd
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

class SimpleGraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pred_len = config['forecast_horizon']
        self.channels = config['channels']
        self.num_nodes = config['num_nodes']
        
        # Node embedding
        self.node_emb = nn.Linear(config['node_dim'], config['channels'] - config['pe_dim'])
        self.pe_lin = nn.Linear(config['pe_walk_length'], config['pe_dim'])
        self.pe_norm = nn.BatchNorm1d(config['pe_walk_length'])
        
        # Edge embedding
        self.edge_emb = nn.Embedding(2, config['channels'])
        
        # GPS Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(config['num_layers']):
            local_gnn = self._create_local_gnn(config)
            
            conv = GPSConv(
                config['channels'], 
                local_gnn, 
                heads=config['num_heads'],
                attn_type='multihead',
                attn_kwargs={'dropout': config['attn_dropout']}
            )
            self.convs.append(conv)
        
        # Normalization
        self.norm = nn.BatchNorm1d(config['channels'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Simplified forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(config['channels'], 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['forecast_horizon'] * config['num_nodes'])
        )

    def _create_local_gnn(self, config):
        """Create local GNN wrapped so it always accepts edge_attr arg"""
        # Build the underlying conv and wrap it so its forward signature is uniform:
        if config.get('local_gnn_type', 'GINE') == 'GINE':
            def create_gine_nn():
                return nn.Sequential(
                    nn.Linear(config['channels'], config['channels']),
                    nn.ReLU(),
                    nn.Linear(config['channels'], config['channels']),
                )
            base_conv = GINEConv(create_gine_nn())
        else:  # GCN
            base_conv = GCNConv(config['channels'], config['channels'])

        # Wrapper ensures forward(x, edge_index, edge_attr=None) always works
        class LocalGNNWrapper(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv
            def forward(self, x, edge_index, edge_attr=None):
                # GINEConv expects edge_attr, GCNConv doesn't — dispatch accordingly
                if isinstance(self.conv, GINEConv):
                    return self.conv(x, edge_index, edge_attr)
                else:
                    # ignore edge_attr for convs that don't use it
                    return self.conv(x, edge_index)

        return LocalGNNWrapper(base_conv)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        # Node embeddings with positional encoding
        x_pe = self.pe_norm(pe)
        node_emb = self.node_emb(x)
        pe_emb = self.pe_lin(x_pe)
        x = torch.cat((node_emb, pe_emb), dim=1)
        
        # Edge attributes for GINE
        edge_attr_emb = self.edge_emb(edge_attr)
        
        # Apply GPS convolutions
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr_emb)
            x = self.norm(x)
            x = self.dropout(x)
        
        # Global pooling and forecasting
        x = global_add_pool(x, batch)
        output = self.forecast_head(x)
        output = output.view(-1, self.pred_len, self.num_nodes)
        return output

# DATA LOADING AND PREPROCESSING

class SimpleDataLoader:
    def __init__(self, data_file, device, seq_len=12, out_len=36, correlation_threshold=0.3):
        self.data_file = data_file
        self.device = device
        self.seq_len = seq_len
        self.out_len = out_len
        
        self.read_data()
        self.create_graph_structure(correlation_threshold)
        self.split_data()
        
    def read_data(self):
        print(f"Loading domain data from: {self.data_file}")

        # READ CSV 
        try:
            df = pd.read_csv(self.data_file, header=0)
            print(f"Loaded {df.shape[1]} columns from header")
        except Exception as e:
            print(f"Error reading with headers: {e}, trying without headers")
            df = pd.read_csv(self.data_file, header=None)
            df.columns = [f"Node_{i}" for i in range(df.shape[1])]
            print(f"Loaded {df.shape[1]} columns without header")

        # FIRST COLUMN IS TIMESTAMP / STRING LABELS
        self.timestamps = df.iloc[:, 0].astype(str).tolist()
        self.dates = self.timestamps   

        timestamp_col = df.columns[0]

        # CONVERT ALL OTHER COLUMNS TO NUMERIC
        numeric_df = df.drop(columns=[timestamp_col]).apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )

        # If no valid numeric columns, skip file safely
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric columns found in CSV")

        # CONVERT TO NUMPY (NUMERIC ONLY!)
        self.raw_data = numeric_df.to_numpy().astype(float)

        print(f"Numeric data shape: {self.raw_data.shape}")

        # NORMALIZATION 
        scale = np.max(np.abs(self.raw_data), axis=0)
        scale[scale == 0] = 1.0

        self.scale = torch.from_numpy(scale).float()
        self.normalized_data = self.raw_data / self.scale.numpy()

        # Metadata 
        self.feature_names = numeric_df.columns.tolist()
        self.n, self.m = self.raw_data.shape
        self.col = self.feature_names

        print(f"Normalized data shape: {self.normalized_data.shape}")
        print(f"Time steps: {self.n}, Nodes: {self.m}")
    
    def create_graph_structure(self, correlation_threshold):
        """Create graph structure based on correlation"""
        print(f"Creating graph structure with threshold={correlation_threshold}")
        
        max_samples = min(100, self.n)
        data_for_corr = self.normalized_data[:max_samples] + 1e-8
        
        try:
            correlation_matrix = np.corrcoef(data_for_corr, rowvar=False)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        except:
            print("Correlation calculation failed, using identity matrix")
            correlation_matrix = np.eye(self.m)
        
        adj_matrix = (np.abs(correlation_matrix) > correlation_threshold).astype(int)
        
        edge_index = []
        edge_attr = []
        
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if adj_matrix[i, j] == 1:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(1)
                    edge_attr.append(1)
        
        if not edge_index:
            print("Warning: No edges found. Creating minimal connected graph.")
            for i in range(min(10, self.m)):
                j = (i + 1) % self.m
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(1)
                edge_attr.append(1)
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        print(f"Graph created: {self.edge_index.shape[1]} edges")
        
    def split_data(self):
        """Create sliding-window sequences from the full series, then split them."""
        total_time = len(self.normalized_data)
        required = self.seq_len + self.out_len
        if total_time < required:
            print("Warning: Not enough total data to form any sequences.")
            self.train = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            self.valid = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            self.test  = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            self.train_dates = []
            self.valid_dates = []
            self.test_dates = []
            return

        # Build all sequences by sliding window across the entire normalized dataset
        n_seq = total_time - required + 1
        X_all = np.zeros((n_seq, self.seq_len, self.m))
        Y_all = np.zeros((n_seq, self.out_len, self.m))
        seq_dates = []  # list of lists containing forecast dates for each sequence
        for i in range(n_seq):
            X_all[i] = self.normalized_data[i:i+self.seq_len]
            Y_all[i] = self.normalized_data[i+self.seq_len:i+self.seq_len+self.out_len]
            # forecast dates for this sequence (use available calendar)
            start_idx = i + self.seq_len
            seq_dates.append(self.dates[start_idx:start_idx + self.out_len])
            seq_dates.append(self.timestamps[start_idx:start_idx + self.out_len])


        # Now split sequences into train/valid/test by ratio
        n_train = int(n_seq * 0.7)
        n_valid = int(n_seq * 0.15)
        n_test  = n_seq - n_train - n_valid

        def to_tensors(arr):
            return torch.from_numpy(arr).float()

        self.train = (to_tensors(X_all[:n_train]), to_tensors(Y_all[:n_train]))
        self.valid = (to_tensors(X_all[n_train:n_train + n_valid]), to_tensors(Y_all[n_train:n_train + n_valid]))
        self.test  = (to_tensors(X_all[n_train + n_valid:]), to_tensors(Y_all[n_train + n_valid:]))

        # Store per-sequence forecast date lists for plotting; keep only sequences that exist
        self.train_dates = seq_dates[:n_train]
        self.valid_dates = seq_dates[n_train:n_train + n_valid]
        self.test_dates  = seq_dates[n_train + n_valid:]

        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")
        
    def _create_sequences(self, data, split_name):
        """Create input-output sequences"""
        available_samples = len(data)
        required_samples = self.seq_len + self.out_len
        
        if available_samples < required_samples:
            print(f"Warning: Not enough {split_name} data for full sequences.")
            return torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m)
        
        n = len(data) - self.seq_len - self.out_len + 1
        if n <= 0:
            return torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m)
            
        X = np.zeros((n, self.seq_len, self.m))
        Y = np.zeros((n, self.out_len, self.m))
        
        for i in range(n):
            X[i] = data[i:i+self.seq_len]
            Y[i] = data[i+self.seq_len:i+self.seq_len+self.out_len]
        
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    
    def prepare_pyg_data(self, sequences, targets, walk_length=20):
        """Convert sequences to PyG Data objects"""
        data_list = []
        
        for seq, target in zip(sequences, targets):
            x = torch.FloatTensor(seq[-1])
            
            transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
            
            # Flatten target
            target_flat = torch.FloatTensor(target).flatten()
            
            data = Data(
                x=x.unsqueeze(1),
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=target_flat,
                num_nodes=self.m
            )
            data = transform(data)
            data_list.append(data)
        
        print(f"Prepared {len(data_list)} PyG data objects")
        return data_list

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
        """Load domain data and create SimpleDataLoader"""
        print(f"\nLoading {self.domain_name} data from {data_path}")
        
        try:
            # Create data loader with domain-specific parameters
            data_loader = SimpleDataLoader(
                data_path, 
                self.device, 
                seq_len=sequence_length, 
                out_len=forecast_horizon,
                correlation_threshold=0.3
            )
            
            print(f"  Data shape: {data_loader.raw_data.shape}")
            print(f"  Features: {data_loader.feature_names}")
            print(f"  Sequences: {len(data_loader.train[0])} train, {len(data_loader.valid[0])} valid")
            
            return data_loader
            
        except Exception as e:
            print(f"  ERROR loading {self.domain_name}: {e}")
            return None
    
    def pretrain(self, data_loader, epochs=100, batch_size=32, learning_rate=0.001):
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
        config['forecast_horizon'] = data_loader.out_len
        
        print(f"  Model config: {config}")
        
        # Create model
        model = self.model_class(config).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Prepare PyG data
        train_data = data_loader.prepare_pyg_data(
            data_loader.train[0].numpy(), 
            data_loader.train[1].numpy()
        )
        
        if len(train_data) == 0:
            print(f"  No training sequences available for {self.domain_name}")
            return None
            
        train_loader = PyGDataLoader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Reshape target to match output
                batch_size_out = output.shape[0]
                target_reshaped = batch.y.view(batch_size_out, data_loader.out_len, data_loader.m)
                
                loss = criterion(output, target_reshaped)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = total_loss / batch_count if batch_count > 0 else total_loss
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            if len(data_loader.valid[0]) > 0:
                valid_data = data_loader.prepare_pyg_data(
                    data_loader.valid[0].numpy(),
                    data_loader.valid[1].numpy()
                )
                valid_loader = PyGDataLoader(valid_data, batch_size=min(batch_size, len(valid_data)), shuffle=False)
                
                with torch.no_grad():
                    for batch in valid_loader:
                        batch = batch.to(self.device)
                        output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                        batch_size_out = output.shape[0]
                        target_reshaped = batch.y.view(batch_size_out, data_loader.out_len, data_loader.m)
                        val_loss += criterion(output, target_reshaped).item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else val_loss
            else:
                avg_val_loss = avg_train_loss  # Use train loss if no validation data
            
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
            
        # Only transfer GPS and GNN layers, not input/output projections
        layers_to_transfer = ['convs', 'norm', 'pe_lin', 'pe_norm']
        layers_to_skip = ['node_emb', 'forecast_head', 'edge_emb']
        
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

# TRAINING AND EVALUATION FUNCTIONS

def train_simple_model(data, model, optimizer, criterion, batch_size, device, epochs):
    """Simplified training function"""
    model.train()
    
    train_data = data.prepare_pyg_data(data.train[0].numpy(), data.train[1].numpy())
    if len(train_data) == 0:
        print("No training data available")
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
            
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Reshape target to match output
            batch_size_out = output.shape[0]
            target_reshaped = batch.y.view(batch_size_out, data.out_len, data.m)
            
            loss = criterion(output, target_reshaped)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count == 0:
            continue
            
        avg_loss = total_loss / batch_count
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Average Loss: {avg_loss:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return best_loss

def evaluate_simple_model(data, model, eval_type, batch_size, device):
    """Comprehensive evaluation function"""
    model.eval()
    
    if eval_type == "Validation":
        X, Y = data.valid
    else:
        X, Y = data.test

    if X.size(0) == 0:
        print(f"No {eval_type} data available")
        return float('inf'), float('inf')

    pyg_data = data.prepare_pyg_data(X.numpy(), Y.numpy())
    
    if len(pyg_data) == 0:
        print(f"No {eval_type} sequences available")
        return float('inf'), float('inf')

    data_loader = PyGDataLoader(pyg_data, batch_size=min(batch_size, len(pyg_data)), shuffle=False)
    
    all_predictions = []
    all_targets = []

    for batch in data_loader:
        batch = batch.to(device)
        
        with torch.no_grad():
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Denormalize predictions and targets
        scale_batch = data.scale.expand(output.size(0), output.size(1), data.m).to(device)
        predictions_denorm = output * scale_batch
        
        # Reshape targets
        batch_size_out = output.shape[0]
        targets_reshaped = batch.y.view(batch_size_out, data.out_len, data.m) * scale_batch

        all_predictions.append(predictions_denorm.cpu())
        all_targets.append(targets_reshaped.cpu())

    if not all_predictions:
        return float('inf'), float('inf')

    try:
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        # Calculate RRSE and RAE
        predict_flat = all_predictions_tensor.flatten()
        test_flat = all_targets_tensor.flatten()
        
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
        
        print(f"{eval_type} - RRSE: {rrse:.6f}, RAE: {rae:.6f}")
        
        return rrse, rae
        
    except Exception as e:
        print(f"Error in {eval_type.lower()} evaluation: {e}")
        return float('inf'), float('inf')

def validate_transfer_learning_benefit(cyber_data, config, device, pretrained_states, epochs=50):
    """
    Compare three scenarios:
    1. Random initialization (baseline)
    2. Pre-trained initialization
    3. Pre-trained + fine-tuned
    
    """
    print(f"\n{'='*80}")
    print("VALIDATING TRANSFER LEARNING BENEFIT")
    print(f"{'='*80}")
    
    results = {}
    
    # Scenario 1: Random initialization
    print("\n1. Training with random initialization...")
    model_random = SimpleGraphTransformer(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model_random.parameters(), lr=0.001, weight_decay=0.01)
    results['random'] = train_simple_model(cyber_data, model_random, optimizer, criterion, config['batch_size'], device, epochs)
    
    # Scenario 2: Pre-trained, frozen backbone
    print("\n2. Training with pre-trained frozen backbone...")
    if pretrained_states:
        transfer_manager = TransferManager(pretrained_states, config, SimpleGraphTransformer, device)
        model_frozen, _ = transfer_manager.create_transferred_model(strategy='selective')
        # Freeze transformer layers
        for name, param in model_frozen.named_parameters():
            if 'convs' in name or 'norm' in name or 'pe_' in name:
                param.requires_grad = False
        optimizer = optim.AdamW(model_frozen.parameters(), lr=0.001, weight_decay=0.01)
        results['pretrained_frozen'] = train_simple_model(cyber_data, model_frozen, optimizer, criterion, config['batch_size'], device, epochs)
    else:
        results['pretrained_frozen'] = float('inf')
    
    # Scenario 3: Pre-trained, fine-tuned (full fine-tuning)
    print("\n3. Training with pre-trained + fine-tuning...")
    if pretrained_states:
        transfer_manager = TransferManager(pretrained_states, config, SimpleGraphTransformer, device)
        model_finetuned, _ = transfer_manager.create_transferred_model(strategy='selective')
        optimizer = optim.AdamW(model_finetuned.parameters(), lr=0.0005, weight_decay=0.01)  # Lower LR for fine-tuning
        results['pretrained_finetuned'] = train_simple_model(cyber_data, model_finetuned, optimizer, criterion, config['batch_size'], device, epochs)
    else:
        results['pretrained_finetuned'] = float('inf')
    
    # Analysis
    print("\nTransfer Learning Validation Results:")
    print(f"Random Init:          Loss = {results['random']:.6f}")
    if results['pretrained_frozen'] != float('inf'):
        print(f"Pretrained (frozen):  Loss = {results['pretrained_frozen']:.6f}")
    if results['pretrained_finetuned'] != float('inf'):
        print(f"Pretrained (tuned):   Loss = {results['pretrained_finetuned']:.6f}")
    
    if results['pretrained_finetuned'] != float('inf') and results['random'] != float('inf'):
        improvement = (results['random'] - results['pretrained_finetuned']) / results['random'] * 100
        print(f"\nImprovement: {improvement:.1f}%")
    
    return results

# MAIN EXECUTION 

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Pre-training for Graph Transformer')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=100, help='epochs for pre-training')
    parser.add_argument('--epochs_finetune', type=int, default=50, help='epochs for initial fine-tuning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--validate_transfer', action='store_true', default=True, help='run transfer learning validation')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs('transfer_learning_pretrain', exist_ok=True)
    
    # Set random seed
    set_random_seed(123)
    
    # Base configuration (fixed hyperparameters for transfer learning)
    base_config = {
        'channels': 64,
        'num_layers': 3,
        'pe_dim': 8,
        'pe_walk_length': 20,
        'node_dim': 1,
        'num_heads': 4,
        'attn_dropout': 0.2,
        'dropout': 0.3,
        'weight_decay': 1e-4,
        'learning_rate': 0.0005,
        'batch_size': 16,
        'forecast_horizon': 6,  # For pre-training
        'sequence_length': 12,  # For pre-training
        'correlation_threshold': 0.3,
        'local_gnn_type': 'GINE',
        'num_nodes': None  # Will be set from data
    }
    
    #  PRE-TRAINING 
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON RELATED DOMAINS")
    print("="*80)
    
    # Find all CSV files in pretrain directory
    pretrain_files = glob.glob(os.path.join(args.pretrain_data_dir, "*.csv"))
    print(f"Found {len(pretrain_files)} pre-training files: {pretrain_files}")
    
    pretrained_models = []
    
    for file_path in pretrain_files:
        domain_name = os.path.splitext(os.path.basename(file_path))[0]
        
        pretrainer = DomainPreTrainer(
            domain_name, SimpleGraphTransformer, base_config, device
        )
        
        data = pretrainer.load_and_prepare_data(
            file_path, 
            sequence_length=base_config['sequence_length'],
            forecast_horizon=base_config['forecast_horizon']
        )
        
        pretrained_state = pretrainer.pretrain(
            data, 
            epochs=args.epochs_pretrain,
            batch_size=base_config['batch_size'],
            learning_rate=base_config['learning_rate']
        )
        
        if pretrained_state is not None:
            pretrained_models.append({
                'domain': domain_name,
                'state_dict': pretrained_state,
                'training_history': pretrainer.training_history
            })
            
            # Save pre-trained model
            torch.save(pretrained_state, f'transfer_learning_pretrain/pretrained_{domain_name}.pt')
            print(f"Saved pre-trained model: transfer_learning_pretrain/pretrained_{domain_name}.pt")
    
    print(f"\nPre-training completed. {len(pretrained_models)} models trained successfully.")
    
    #  TRANSFER LEARNING VALIDATION 
    if args.validate_transfer and pretrained_models:
        print("\nLoading cyber threat data for validation...")
        try:
            # Load cyber threat data with shorter horizon for validation
            cyber_data = SimpleDataLoader(
                args.cyber_data, 
                device, 
                seq_len=12, 
                out_len=6,  # Shorter horizon for validation
                correlation_threshold=0.3
            )
            
            # Cyber threat model config
            cyber_config = base_config.copy()
            cyber_config['num_nodes'] = cyber_data.m
            cyber_config['forecast_horizon'] = cyber_data.out_len
            
            # Run validation
            results = validate_transfer_learning_benefit(
                cyber_data, cyber_config, device, pretrained_models, epochs=30
            )
            
            # Save validation results
            with open('transfer_learning_pretrain/transfer_validation.json', 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            print(f"Error during transfer learning validation: {e}")
    
    #  TRANSFER + INITIAL FINE-TUNING 
    if pretrained_models:
        print("\n" + "="*80)
        print("PHASE 2: TRANSFER + INITIAL FINE-TUNING")
        print("="*80)
        
        # Load cyber threat data with operational horizon
        print("Loading cyber threat data for fine-tuning...")
        cyber_data = SimpleDataLoader(
            args.cyber_data, 
            device, 
            seq_len=18,  # Operational sequence length
            out_len=36,  # Operational forecast horizon
            correlation_threshold=0.3
        )
        
        # Cyber threat model config
        cyber_config = base_config.copy()
        cyber_config['num_nodes'] = cyber_data.m
        cyber_config['forecast_horizon'] = cyber_data.out_len
        cyber_config['sequence_length'] = cyber_data.seq_len
        
        # Transfer weights
        transfer_manager = TransferManager(
            pretrained_models, cyber_config, SimpleGraphTransformer, device
        )
        
        transferred_model, transferred_layers = transfer_manager.create_transferred_model(
            strategy='selective'
        )
        
        # Initial fine-tuning
        print(f"\nInitial fine-tuning for {args.epochs_finetune} epochs...")
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(transferred_model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower LR for fine-tuning
        
        best_val_loss = train_simple_model(
            cyber_data, transferred_model, optimizer, criterion, 
            cyber_config['batch_size'], device, args.epochs_finetune
        )
        
        # Evaluate the fine-tuned model
        val_rrse, val_rae = evaluate_simple_model(
            cyber_data, transferred_model, "Validation", 
            cyber_config['batch_size'], device
        )
        
        # Save the fine-tuned model for further hyperparameter optimization
        torch.save({
            'model_state_dict': transferred_model.state_dict(),
            'config': cyber_config,
            'transferred_layers': transferred_layers,
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'fine_tune_val_loss': best_val_loss,
            'val_rrse': val_rrse,
            'val_rae': val_rae
        }, 'transfer_learning_pretrain/transferred_model_for_hp_search.pt')
        
        print(f"\nPhase 2 completed. Fine-tuned model saved for hyperparameter optimization.")
        print(f"Final validation loss: {best_val_loss:.6f}")
        print(f"Validation RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
    
    print(f"\nPre-training and transfer learning pipeline completed.")
    print(f"Results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()