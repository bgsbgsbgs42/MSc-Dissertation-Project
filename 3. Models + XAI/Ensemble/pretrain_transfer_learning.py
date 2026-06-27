import argparse
import math
import time
import torch
import torch.nn as nn
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

# MODEL ARCHITECTURE 
class CyberThreatDataset(Dataset):
    def __init__(self, data, sequence_length=12, forecast_horizon=36):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.data = data.values
        else:
            self.data = data
            
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.num_nodes = self.data.shape[1]
        
    def __len__(self):
        total_length = len(self.data) - self.sequence_length - self.forecast_horizon + 1
        return max(0, total_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Ensure patch_size is compatible with sequence length
        if patch_size > seq_len:
            patch_size = seq_len
            print(f"Warning: patch_size > seq_len. Using patch_size = {patch_size}")
        
        # Find compatible patch size
        divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0 and i <= patch_size]
        if divisors:
            patch_size = max(divisors)
        else:
            # Find the closest divisor
            for i in range(patch_size, 0, -1):
                if seq_len % i == 0:
                    patch_size = i
                    break
            else:
                patch_size = 1
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        print(f"PatchEmbedding: seq_len={seq_len}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Proper 1D convolution for time series
        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Initialize with smaller values for stability
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Dynamic position embeddings based on actual num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes)
        batch_size = x.shape[0]
        
        # Reshape for conv1d: (batch_size, num_nodes, seq_len)
        x = x.transpose(1, 2)
        
        # Apply 1D convolution: (batch_size, embed_dim, num_patches)
        x = self.projection(x)
        
        # Transpose back: (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings - sizes are guaranteed to match
        x = x + self.position_embeddings
        
        return x

class VisionTransformerForTimeSeries(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #  Validate and adjust config before creating layers
        self._validate_and_adjust_config(config)
        
        actual_patch_size = config['patch_size']
        actual_num_patches = config['sequence_length'] // actual_patch_size
        
        print(f"ViT config: sequence_length={config['sequence_length']}, patch_size={actual_patch_size}, num_patches={actual_num_patches}, forecast_horizon={config['forecast_horizon']}")
        
        self.patch_embed = PatchEmbedding(
            seq_len=config['sequence_length'],
            patch_size=actual_patch_size,
            in_channels=config['num_nodes'],
            embed_dim=config['embed_dim']
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_dim'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
        # Initialize forecast head with smaller weights
        self.forecast_head = nn.Sequential(
            nn.Linear(config['embed_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['forecast_horizon'] * config['num_nodes'])
        )
        
        # Initialize weights properly
        self._init_weights()
        
        self.mc_dropout = nn.Dropout(config['mc_dropout'])
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
    def _validate_and_adjust_config(self, config):
        """ Config validation and adjustment"""
        config['sequence_length'] = int(config['sequence_length'])
        config['patch_size'] = int(config['patch_size'])
        config['embed_dim'] = int(config['embed_dim'])
        config['num_heads'] = int(config['num_heads'])
        config['hidden_dim'] = int(config['hidden_dim'])
        config['num_layers'] = int(config['num_layers'])
        config['batch_size'] = int(config['batch_size'])
        config['forecast_horizon'] = int(config['forecast_horizon'])
        config['num_nodes'] = int(config['num_nodes'])
        
        # Ensure patch_size is compatible with sequence_length
        if config['patch_size'] > config['sequence_length']:
            config['patch_size'] = config['sequence_length']
            print(f"Adjusted patch_size to sequence_length: {config['patch_size']}")
        
        # Find compatible patch size
        if config['sequence_length'] % config['patch_size'] != 0:
            divisors = [i for i in range(1, config['sequence_length'] + 1) 
                       if config['sequence_length'] % i == 0 and i <= config['patch_size']]
            if divisors:
                config['patch_size'] = max(divisors)
                print(f"Adjusted patch_size to compatible value: {config['patch_size']}")
            else:
                # Find the largest divisor
                for i in range(config['patch_size'], 0, -1):
                    if config['sequence_length'] % i == 0:
                        config['patch_size'] = i
                        break
                else:
                    config['patch_size'] = 1
                print(f"Using compatible patch_size: {config['patch_size']}")
        
        # Ensure num_heads is compatible with embed_dim
        if config['embed_dim'] % config['num_heads'] != 0:
            divisors = [i for i in range(1, config['embed_dim'] + 1) 
                       if config['embed_dim'] % i == 0 and i <= config['num_heads']]
            if divisors:
                config['num_heads'] = max(divisors)
                print(f"Adjusted num_heads to compatible value: {config['num_heads']}")
            else:
                config['num_heads'] = 1
                print(f"Using num_heads: 1")
    
    def forward(self, x, mc_dropout=True):
        # x shape: (batch_size, seq_len, num_nodes)
        x = self.patch_embed(x)
        
        x = self.transformer(x)
        
        cls_token = x[:, 0]
        
        if mc_dropout:
            cls_token = self.mc_dropout(cls_token)
            
        forecast = self.forecast_head(cls_token)
        forecast = forecast.view(-1, self.config['forecast_horizon'], self.config['num_nodes'])
        
        return forecast

class GraphAwareEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        x = self.node_proj(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
            x = self.dropout(x)
        return x

class SpatioTemporalEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create ViT config with validated parameters
        vit_config = {
            'sequence_length': config['sequence_length'],
            'patch_size': config['vit_patch_size'],
            'embed_dim': config['vit_embed_dim'],
            'num_heads': config['vit_num_heads'],
            'hidden_dim': config['vit_hidden_dim'],
            'num_layers': config['vit_num_layers'],
            'dropout': config['dropout'],
            'mc_dropout': config['mc_dropout'],
            'forecast_horizon': config['forecast_horizon'],  #  Use the same forecast horizon
            'num_nodes': config['num_nodes'],
            'batch_size': config.get('batch_size', 4)
        }
        
        # Vision Transformer branch (temporal patterns)
        self.vit_branch = VisionTransformerForTimeSeries(vit_config)
        
        # Graph-aware branch (relational patterns)
        self.graph_branch = GraphAwareEncoder(
            node_dim=config['num_nodes'],
            hidden_dim=config['graph_hidden_dim'],
            num_layers=config['graph_num_layers'],
            dropout=config['dropout']
        )
        
        # Cross-attention fusion module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config['vit_embed_dim'],
            num_heads=4,
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Dynamic weighting mechanism
        self.attention_weights = nn.Parameter(torch.ones(2))
        self.ensemble_dropout = nn.Dropout(config['mc_dropout'])
        
        # Enhanced forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(config['vit_embed_dim'] + config['graph_hidden_dim'], config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, config['forecast_horizon'] * config['num_nodes'])
        )
        
    def forward(self, x, mc_dropout=True):
        batch_size, seq_len, num_nodes = x.shape
        
        # Vision Transformer branch
        vit_features = self.vit_branch.patch_embed(x)
        vit_features = self.vit_branch.transformer(vit_features)
        vit_cls = vit_features[:, 0]  # CLS token
        
        # Graph-aware branch (treat time series as graph)
        graph_features = self.graph_branch(x)
        graph_global = graph_features.mean(dim=1)  # Global pooling
        
        # Dynamic weighting
        weights = torch.softmax(self.attention_weights, dim=0)
        weighted_vit = weights[0] * vit_cls
        weighted_graph = weights[1] * graph_global
        
        # Feature fusion with cross-attention
        fused_features = torch.cat([weighted_vit, weighted_graph], dim=1)
        
        # Apply MC dropout for uncertainty
        if mc_dropout:
            fused_features = self.ensemble_dropout(fused_features)
            
        # Final forecast
        forecast = self.forecast_head(fused_features)
        forecast = forecast.view(batch_size, self.config['forecast_horizon'], self.config['num_nodes'])
        
        return forecast

# DATA LOADERS FOR TRANSFER LEARNING

class DomainDataset(Dataset):
    """Dataset for domain-specific time series data for pre-training"""
    def __init__(self, data, sequence_length=12, forecast_horizon=6):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.data = data.values
        else:
            self.data = data
            
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
            
        self.num_nodes = self.data.shape[1]
        
    def __len__(self):
        total_length = len(self.data) - self.sequence_length - self.forecast_horizon + 1
        return max(0, total_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class DataLoaderEnsemble:
    """Data loader for cyber threat data - same as original"""
    def __init__(self, data_file, train_ratio, valid_ratio, device, horizon, seq_len, normalize, out_len):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.device = device
        self.horizon = horizon
        self.seq_len = seq_len
        self.normalize = normalize
        self.out_len = out_len  # This should match forecast_horizon
        
        self.P = seq_len
        self.h = horizon
        
        self.read_data()
        self.split_data()
        
    def read_data(self):
        print(f"Loading data from: {self.data_file}")

        # Load CSV with headers
        self.data = pd.read_csv(self.data_file)
        self.raw_data = self.data.values
        print(f"Raw data shape: {self.raw_data.shape}")

        # Use actual column headers as feature names
        self.feature_names = list(self.data.columns)
        self.n, self.m = self.raw_data.shape
        self.col = self.feature_names  

        # Normalize data
        if self.normalize == 2:
            self.scale = torch.from_numpy(np.max(np.abs(self.raw_data), axis=0)).float()
            self.scale[self.scale == 0] = 1.0
        else:
            self.scale = torch.ones(self.m).float()
            
        self.normalized_data = self.raw_data / self.scale.numpy()
        print(f"Normalized data shape: {self.normalized_data.shape}")
        print(f"Feature names loaded: {', '.join(self.feature_names[:10])}{'...' if self.m > 10 else ''}")

        
    def split_data(self):
        """Split data into train, validation, and test sets """
        total_samples = len(self.normalized_data)
        
        # Calculate split points - adjusted for proper validation period
        n_train = int(total_samples * self.train_ratio)
        n_valid = int(total_samples * self.valid_ratio)
        
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {n_train}, Validation samples: {n_valid}")
        print(f"Sequence length: {self.seq_len}, Output length: {self.out_len}")
        
        # Split data
        train_data = self.normalized_data[:n_train]
        valid_data = self.normalized_data[n_train:n_train + n_valid]
        test_data = self.normalized_data[n_train + n_valid:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Create sequences for each split
        self.train = self._batchify(train_data, self.seq_len, self.out_len)
        self.valid = self._batchify(valid_data, self.seq_len, self.out_len)
        self.test = self._batchify(test_data, self.seq_len, self.out_len)
        
        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")
        
        # Create test window for sliding window evaluation
        self.test_window = torch.from_numpy(test_data).float().to(self.device)
        
    def _batchify(self, data, seq_len, out_len):
        """Create sequences from data"""
        n = len(data) - seq_len - out_len + 1
        if n <= 0:
            print(f"Warning: Not enough data to create sequences. Need {seq_len + out_len} samples, but have {len(data)}")
            return torch.zeros(0, seq_len, self.m), torch.zeros(0, out_len, self.m)
            
        X = np.zeros((n, seq_len, self.m))
        Y = np.zeros((n, out_len, self.m))
        
        for i in range(n):
            X[i] = data[i:i+seq_len]
            Y[i] = data[i+seq_len:i+seq_len+out_len]
            
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    
    def get_batches(self, X, Y, batch_size, shuffle):
        """Generate batches from data"""
        if X.size(0) == 0 or Y.size(0) == 0:
            print(f"Warning: No data to create batches. X shape: {X.shape}, Y shape: {Y.shape}")
            return
            
        if shuffle:
            indices = torch.randperm(X.size(0))
            X = X[indices]
            Y = Y[indices]
            
        for i in range(0, X.size(0), batch_size):
            yield X[i:i+batch_size], Y[i:i+batch_size]

# TRANSFER LEARNING COMPONENTS

class DomainPreTrainer:
    """Handles pre-training on a single domain dataset"""
    
    def __init__(self, domain_name, model_class, base_config, device):
        self.domain_name = domain_name
        self.model_class = model_class
        self.base_config = base_config.copy()
        self.device = device
        self.best_state = None
        self.training_history = []
    
    def load_and_prepare_data(self, data_path, sequence_length=12, forecast_horizon=6):
        """Load domain data and create sequences for pre-training"""
        print(f"\nLoading {self.domain_name} data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Remove date/timestamp columns and keep only numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        print(f"  Raw shape: {data.shape}")
        print(f"  Features: {list(numeric_cols)}")
        
        # Handle single-column data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Normalize the data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1.0
        data_normalized = (data - data_mean) / data_std
        
        # Create dataset
        dataset = DomainDataset(data_normalized, sequence_length, forecast_horizon)
        
        if len(dataset) == 0:
            print(f"  WARNING: No sequences could be created from this data!")
            return None
        
        # Create sequences using the dataset
        sequences = []
        targets = []
        
        for i in range(len(dataset)):
            x, y = dataset[i]
            sequences.append(x.numpy())
            targets.append(y.numpy())
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  Created {len(sequences)} sequences")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return {
            'X_train': torch.FloatTensor(X_train),
            'y_train': torch.FloatTensor(y_train),
            'X_val': torch.FloatTensor(X_val),
            'y_val': torch.FloatTensor(y_val),
            'num_features': data.shape[1],
            'mean': data_mean,
            'std': data_std,
            'dataset': dataset
        }
    
    def pretrain(self, data_dict, epochs=100, batch_size=32, learning_rate=0.001):
        """Pre-train model on this domain"""
        if data_dict is None:
            print(f"  SKIPPING {self.domain_name} - no valid data")
            return None
            
        print(f"\n{'='*80}")
        print(f"PRE-TRAINING ON {self.domain_name.upper()}")
        print(f"{'='*80}")
        
        # Adjust config for this domain
        config = self.base_config.copy()
        config['num_nodes'] = data_dict['num_features']
        config['forecast_horizon'] = data_dict['y_train'].shape[1]
        
        print(f"  Model config: {config}")
        
        # Create model 
        model = self.model_class(config).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            n_batches = 0
            
            # Simple batching with shuffling
            indices = torch.randperm(len(data_dict['X_train']))
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = data_dict['X_train'][batch_indices].to(self.device)
                y_batch = data_dict['y_train'][batch_indices].to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else train_loss
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(data_dict['X_val'].to(self.device))
                val_loss = criterion(val_output, data_dict['y_val'].to(self.device))
            
            scheduler.step(val_loss)
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss.item()
            })
            
            # Early stopping and best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                self.best_state = model.state_dict().copy()
                patience_counter = 0
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={val_loss.item():.6f} *")
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={val_loss.item():.6f}")
            
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"\nPre-training completed. Best val loss: {best_val_loss:.6f}")
        return self.best_state

class TransferManager:
    """Manages transfer of pre-trained weights to the ensemble model"""
    
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
        layers_to_transfer = ['transformer', 'patch_embed.projection', 'position_embeddings']
        layers_to_skip = ['forecast_head', 'cls_token', 'graph_branch', 'cross_attention', 'ensemble_dropout']
        
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

# TRAINING UTILITIES

def train_stable(data, X, Y, model, criterion, optimizer, batch_size, device):
    """Stable training function with gradient monitoring - same as original"""
    model.train()
    total_loss = 0
    n_samples = 0

    if X.size(0) == 0:
        print("CRITICAL: No training data available")
        return float('inf')

    batch_count = 0
    for batch_idx, (X_batch, Y_batch) in enumerate(data.get_batches(X, Y, batch_size, True)):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        
        # Ensure output and Y_batch have compatible dimensions
        model_forecast_horizon = output.shape[1]
        data_forecast_horizon = Y_batch.shape[1]
        
        if model_forecast_horizon != data_forecast_horizon:
            print(f"Training: Model horizon ({model_forecast_horizon}) != Data horizon ({data_forecast_horizon})")
            # Use the minimum of both
            actual_forecast_horizon = min(model_forecast_horizon, data_forecast_horizon)
            output = output[:, :actual_forecast_horizon, :]
            Y_batch = Y_batch[:, :actual_forecast_horizon, :]
        
        scale = data.scale.expand(output.size(0), output.size(1), data.m).to(device)
        output = output * scale
        Y_batch = Y_batch * scale

        loss = criterion(output, Y_batch)
        loss.backward()
        
        # Monitor gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Adaptive gradient clipping
        clip_value = max(0.1, min(1.0, total_norm))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()

        total_loss += loss.item()
        n_samples += (output.size(0) * output.size(1) * data.m)
        batch_count += 1
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}, Grad Norm: {total_norm:.6f}")

    avg_loss = total_loss / n_samples if n_samples > 0 else total_loss
    print(f"  Training completed: {batch_count} batches, Avg Loss: {avg_loss:.6f}")
    return avg_loss

def create_directories():
    """Create all necessary directories for transfer learning"""
    directories = [
        'transfer_learning_pretrain',
        'transfer_learning_pretrain/pretrained_models',
        'transfer_learning_pretrain/results',
        'transfer_learning_pretrain/plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Pre-training and Fine-tuning')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=100, help='epochs for pre-training each domain')
    parser.add_argument('--epochs_finetune', type=int, default=50, help='epochs for initial fine-tuning')
    parser.add_argument('--batch_size_pretrain', type=int, default=32, help='batch size for pre-training')
    parser.add_argument('--batch_size_finetune', type=int, default=8, help='batch size for fine-tuning')
    parser.add_argument('--learning_rate_pretrain', type=float, default=0.001, help='learning rate for pre-training')
    parser.add_argument('--learning_rate_finetune', type=float, default=0.0005, help='learning rate for fine-tuning')
    parser.add_argument('--transfer_strategy', type=str, default='ensemble', choices=['single', 'ensemble', 'selective'], help='strategy for weight transfer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create output directories
    create_directories()
    
    # Set random seed
    set_random_seed(123)
    
    # Base configuration for pre-training 
    base_pretrain_config = {
        'sequence_length': 12,
        'patch_size': 4,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'mc_dropout': 0.2,
        'forecast_horizon': 6,  # Shorter horizon for pre-training
        'batch_size': args.batch_size_pretrain
    }
    
    # Configuration for cyber threat model 
    cyber_config = {
        'sequence_length': 12,
        'forecast_horizon': 12,  # Adjust based on data availability
        
        # Vision Transformer branch
        'vit_patch_size': 4,
        'vit_embed_dim': 128,
        'vit_num_heads': 8,
        'vit_hidden_dim': 256,
        'vit_num_layers': 3,
        
        # Graph-aware branch
        'graph_hidden_dim': 128,
        'graph_num_layers': 2,
        
        # Fusion mechanism
        'fusion_dim': 256,
        
        # Regularization
        'dropout': 0.2,
        'mc_dropout': 0.2,
        
        # Optimization
        'learning_rate': args.learning_rate_finetune,
        'weight_decay': 0.01,
        'batch_size': args.batch_size_finetune
    }
    
    # PHASE 1: PRE-TRAINING ON RELATED DOMAINS
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
            domain_name, VisionTransformerForTimeSeries, base_pretrain_config, device
        )
        
        data = pretrainer.load_and_prepare_data(file_path)
        pretrained_state = pretrainer.pretrain(
            data, 
            epochs=args.epochs_pretrain,
            batch_size=args.batch_size_pretrain,
            learning_rate=args.learning_rate_pretrain
        )
        
        if pretrained_state is not None:
            pretrained_models.append({
                'domain': domain_name,
                'state_dict': pretrained_state,
                'training_history': pretrainer.training_history
            })
            
            # Save pre-trained model
            model_path = f'transfer_learning_pretrain/pretrained_models/pretrained_{domain_name}.pt'
            torch.save({
                'state_dict': pretrained_state,
                'config': base_pretrain_config,
                'domain': domain_name,
                'training_history': pretrainer.training_history
            }, model_path)
            print(f"Saved pre-trained model: {model_path}")
    
    print(f"\nPre-training completed. {len(pretrained_models)} models trained successfully.")
    
    # Save pre-training summary
    if pretrained_models:
        pretrain_summary = {
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'total_models': len(pretrained_models),
            'pretrain_config': base_pretrain_config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('transfer_learning_pretrain/results/pretraining_summary.json', 'w') as f:
            json.dump(pretrain_summary, f, indent=2, cls=NumpyEncoder)
    
    # PHASE 2: TRANSFER + INITIAL FINE-TUNING 
    if pretrained_models:
        print("\n" + "="*80)
        print("PHASE 2: TRANSFER + INITIAL FINE-TUNING")
        print("="*80)
        
        # Load cyber threat data
        print("Loading cyber threat data for fine-tuning...")
        try:
            cyber_data_loader = DataLoaderEnsemble(
                args.cyber_data, 0.6, 0.2, device, 
                horizon=1, seq_len=12, normalize=2, out_len=12
            )
            
            # Update cyber config with actual data dimensions
            cyber_config['num_nodes'] = cyber_data_loader.m
            cyber_config['forecast_horizon'] = cyber_data_loader.out_len
            
            print(f"Cyber threat data loaded: {cyber_data_loader.m} nodes, {cyber_data_loader.out_len} forecast horizon")
            print(f"Training sequences: {cyber_data_loader.train[0].shape[0]}")
            print(f"Validation sequences: {cyber_data_loader.valid[0].shape[0]}")
            
        except Exception as e:
            print(f"Error loading cyber threat data: {e}")
            return
        
        # Transfer weights to ensemble model
        transfer_manager = TransferManager(
            pretrained_models, cyber_config, SpatioTemporalEnsemble, device
        )
        
        transferred_model, transferred_layers = transfer_manager.create_transferred_model(
            strategy=args.transfer_strategy
        )
        
        # Initial fine-tuning
        print(f"\nInitial fine-tuning for {args.epochs_finetune} epochs...")
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(transferred_model.parameters(), lr=args.learning_rate_finetune, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        best_state = None
        training_history = []
        
        for epoch in range(1, args.epochs_finetune + 1):
            # Training
            train_loss = train_stable(
                cyber_data_loader, cyber_data_loader.train[0], cyber_data_loader.train[1],
                transferred_model, criterion, optimizer, cyber_config['batch_size'], device
            )
            
            # Validation
            transferred_model.eval()
            with torch.no_grad():
                val_output = transferred_model(cyber_data_loader.valid[0].to(device))
                val_loss = criterion(val_output, cyber_data_loader.valid[1].to(device))
            
            scheduler.step(val_loss)
            
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss.item()
            })
            
            print(f"Fine-tune Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss.item():.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = transferred_model.state_dict().copy()
        
        # Save the fine-tuned model
        model_save_data = {
            'model_state_dict': best_state,
            'config': cyber_config,
            'transferred_layers': transferred_layers,
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'transfer_strategy': args.transfer_strategy,
            'fine_tune_val_loss': best_val_loss,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_save_data, 'transfer_learning_pretrain/transferred_model_for_hp_search.pt')
        
        # Save fine-tuning results
        results = {
            'final_val_loss': best_val_loss,
            'transferred_layers_count': len(transferred_layers),
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'transfer_strategy': args.transfer_strategy,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('transfer_learning_pretrain/results/fine_tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nPhase 2 completed successfully!")
        print(f"Final validation loss: {best_val_loss:.6f}")
        print(f"Transferred {len(transferred_layers)} layers using {args.transfer_strategy} strategy")
        print(f"Fine-tuned model saved for hyperparameter optimization")
    
    else:
        print("\nNo pre-trained models available. Skipping Phase 2.")
    
    print(f"\nTransfer learning pipeline completed.")
    print(f"All results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()