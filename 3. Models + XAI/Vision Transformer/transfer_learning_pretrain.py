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

from traitlets import Bool

plt.rcParams['savefig.dpi'] = 1200

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom JSON encoder to handle numpy data types
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
class PatchEmbedding(nn.Module):
    """Convert time series into patches with proper size handling """
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Ensure sequence length is divisible by patch size
        if seq_len % patch_size != 0:
            # Find the largest patch size that divides sequence length
            divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
            if divisors:
                patch_size = max(divisors)  # Use largest divisor for stability
                print(f"Adjusted patch_size to: {patch_size} (divisor of {seq_len})")
            else:
                patch_size = 1  # Fallback
                print(f"Using fallback patch_size: 1")
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        print(f"Final: seq_len={seq_len}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Proper 1D convolution for time series
        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Initialize with smaller values for stability
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Position embeddings match actual num_patches
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
        
        # Add position embeddings - sizes should match now
        x = x + self.position_embeddings
        
        return x

class VisionTransformerForTimeSeries(nn.Module):
    """Vision Transformer adapted for time series forecasting """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self._validate_config(config)
        
        # Calculate actual patch size and num_patches before creating patch_embed
        actual_patch_size = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
        actual_num_patches = config['sequence_length'] // actual_patch_size
        
        print(f"Model config: sequence_length={config['sequence_length']}, patch_size={actual_patch_size}, num_patches={actual_num_patches}, forecast_horizon={config['forecast_horizon']}")
        
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
        
    def _validate_config(self, config):
        config['sequence_length'] = int(config['sequence_length'])
        config['patch_size'] = int(config['patch_size'])
        config['embed_dim'] = int(config['embed_dim'])
        config['num_heads'] = int(config['num_heads'])
        config['hidden_dim'] = int(config['hidden_dim'])
        config['num_layers'] = int(config['num_layers'])
        config['batch_size'] = int(config['batch_size'])
        config['forecast_horizon'] = int(config['forecast_horizon'])
        config['num_nodes'] = int(config['num_nodes'])
        
        # Ensure compatibility before creating layers
        if config['sequence_length'] % config['patch_size'] != 0:
            config['patch_size'] = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
            print(f"Validated patch_size: {config['patch_size']}")
            
        if config['embed_dim'] % config['num_heads'] != 0:
            config['num_heads'] = self._get_compatible_heads(config['embed_dim'], config['num_heads'])
            print(f"Validated num_heads: {config['num_heads']}")
    
    def _get_compatible_patch_size(self, seq_len, desired_patch_size):
        """Find compatible patch size that divides sequence length"""
        divisors = []
        for i in range(1, seq_len + 1):
            if seq_len % i == 0:
                divisors.append(i)
        
        if not divisors:
            return 1  # Fallback
        
        # Use the largest divisor for stability (not the closest)
        return max(divisors)
    
    def _get_compatible_heads(self, embed_dim, desired_heads):
        """Find compatible number of heads that divides embed_dim"""
        divisors = []
        for i in range(1, embed_dim + 1):
            if embed_dim % i == 0:
                divisors.append(i)
        
        if not divisors:
            return 1  # Fallback
        
        # Prefer smaller number of heads for stability
        compatible_heads = [h for h in divisors if h <= min(desired_heads, 16)]
        if compatible_heads:
            return max(compatible_heads)
        else:
            return min(divisors)
        
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
class DomainDataset(Dataset):
    """Dataset for domain-specific time series data"""
    def __init__(self, data, sequence_length=12, forecast_horizon=6, frequency='monthly'):
        """
        Args:
            data: numpy array or pandas DataFrame
            sequence_length: input sequence length
            forecast_horizon: output forecast horizon
            frequency: 'daily' or 'monthly' - affects how we handle the data
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.data = data.values
        else:
            self.data = data
            
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.frequency = frequency
        
        # For daily data, we might need to aggregate or sample
        if frequency == 'daily' and self.data.shape[0] > 365 * 5:  # More than 5 years of daily data
            # Sample to get manageable sequence length
            sampling_rate = max(1, self.data.shape[0] // (365 * 3))  # Target ~3 years worth
            indices = np.arange(0, self.data.shape[0], sampling_rate)
            self.data = self.data[indices]
            print(f"Sampled daily data from {self.data.shape[0] * sampling_rate} to {self.data.shape[0]} points")
        
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

class CyberThreatDataset(Dataset):
    """Dataset for cyber threat time series data"""
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

class DataLoaderS:
    """Data loader for time series data - for flexible forecasting"""
    def __init__(self, data_file, train_ratio, valid_ratio, device, horizon, seq_len, normalize, out_len):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.device = device
        self.horizon = horizon
        self.seq_len = seq_len
        self.normalize = normalize
        self.out_len = out_len
        
        self.P = seq_len
        self.h = horizon
        
        self.read_data()
        self.split_data()
        
    def read_data(self): 
        """Read and normalize data"""
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
        """Split data into train, validation, and test sets - for shorter sequences"""
        total_samples = len(self.normalized_data)
        
        # Calculate split points
        n_train = int(total_samples * self.train_ratio)
        n_valid = int(total_samples * self.valid_ratio)
        
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {n_train}, Validation samples: {n_valid}")
        print(f"Sequence length: {self.seq_len}, Forecast horizon: {self.out_len}")
        
        # Split data
        train_data = self.normalized_data[:n_train]
        valid_data = self.normalized_data[n_train:n_train + n_valid]
        test_data = self.normalized_data[n_train + n_valid:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Ahorter forecast horizon for validation and test if needed
        train_out_len = self.out_len  # Use full horizon for training
        
        # For validation and test, use shorter horizon if needed
        valid_out_len = min(self.out_len, len(valid_data) - self.seq_len)
        test_out_len = min(self.out_len, len(test_data) - self.seq_len)
        
        # Ensure we have at least 1 sequence
        valid_out_len = max(1, valid_out_len)
        test_out_len = max(1, test_out_len)
        
        print(f"Adjusted horizons - Train: {train_out_len}, Validation: {valid_out_len}, Test: {test_out_len}")
        
        # Create sequences for each split
        self.train = self._batchify(train_data, self.seq_len, train_out_len)
        self.valid = self._batchify(valid_data, self.seq_len, valid_out_len)
        self.test = self._batchify(test_data, self.seq_len, test_out_len)
        
        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")
        
        # Store the actual horizons used
        self.train_horizon = train_out_len
        self.valid_horizon = valid_out_len
        self.test_horizon = test_out_len
        
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

#
# TRANSFER LEARNING COMPONENTS
#

class DomainPreTrainer:
    """Handles pre-training on a single domain"""
    
    def __init__(self, domain_name, model_class, base_config, device):
        self.domain_name = domain_name
        self.model_class = model_class
        self.base_config = base_config.copy()
        self.device = device
        self.best_state = None
        self.training_history = []
    
    def load_and_prepare_data(self, data_path, sequence_length=12, frequency='monthly'):
        """Load domain data and create sequences"""
        print(f"\nLoading {self.domain_name} data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Remove date/timestamp columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        print(f"  Raw shape: {data.shape}")
        print(f"  Features: {list(numeric_cols)}")
        print(f"  Frequency: {frequency}")
        
        # Create dataset
        dataset = DomainDataset(data, sequence_length, forecast_horizon=6, frequency=frequency)
        
        if len(dataset) == 0:
            print(f"  WARNING: No sequences could be created from this data!")
            return None
        
        # Normalize
        data_normalized = dataset.data
        data_mean = np.mean(data_normalized, axis=0)
        data_std = np.std(data_normalized, axis=0)
        data_std[data_std == 0] = 1.0
        data_normalized = (data_normalized - data_mean) / data_std
        
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
        
        # Split into train/val
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
            
            # Simple batching
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
            
            # Early stopping
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
            
            if patience_counter >= 50:
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
            
        # Only transfer transformer encoder layers, not input/output projections
        layers_to_transfer = ['transformer', 'position_embeddings']
        layers_to_skip = ['patch_embed.projection', 'forecast_head', 'cls_token']
        
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
    model_random = VisionTransformerForTimeSeries(config).to(device)
    results['random'] = train_and_evaluate_simple(model_random, cyber_data, epochs=epochs, device=device)
    
    # Scenario 2: Pre-trained, frozen backbone
    print("\n2. Training with pre-trained frozen backbone...")
    if pretrained_states:
        transfer_manager = TransferManager(pretrained_states, config, VisionTransformerForTimeSeries, device)
        model_frozen, _ = transfer_manager.create_transferred_model(strategy='selective')
        # Freeze transformer layers
        for name, param in model_frozen.named_parameters():
            if 'transformer' in name or 'patch_embed' in name:
                param.requires_grad = False
        results['pretrained_frozen'] = train_and_evaluate_simple(model_frozen, cyber_data, epochs=epochs, device=device)
    else:
        results['pretrained_frozen'] = float('inf')
    
    # Scenario 3: Pre-trained, fine-tuned (full fine-tuning)
    print("\n3. Training with pre-trained + fine-tuning...")
    if pretrained_states:
        transfer_manager = TransferManager(pretrained_states, config, VisionTransformerForTimeSeries, device)
        model_finetuned, _ = transfer_manager.create_transferred_model(strategy='selective')
        results['pretrained_finetuned'] = train_and_evaluate_simple(model_finetuned, cyber_data, epochs=epochs, device=device)
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

def train_and_evaluate_simple(model, data, epochs, device):
    """Simple training and evaluation for transfer learning validation"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        
        indices = torch.randperm(len(data['X_train']))
        batch_size = 32
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = data['X_train'][batch_indices].to(device)
            y_batch = data['y_train'][batch_indices].to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(data['X_val'].to(device))
            val_loss = criterion(val_output, data['y_val'].to(device))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Pre-training')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=100, help='epochs for pre-training')
    parser.add_argument('--epochs_finetune', type=int, default=20, help='epochs for initial fine-tuning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--validate_transfer', default=True, action='store_true', help='run transfer learning validation')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs('transfer_learning_pretrain', exist_ok=True)
    
    # Set random seed
    set_random_seed(123)
    
    # Base configuration
    base_config = {
        'sequence_length': 12,
        'patch_size': 4,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'mc_dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 8,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'positional_encoding_type': 'learned',
        'mc_inference_runs': 20,
        'forecast_horizon': 6,  # For pre-training
        'grad_clip_value': 1.0,
        'activation': 'relu'
    }
    
    #  PRE-TRAINING ==
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON RELATED DOMAINS")
    print("="*80)
    
    # Find all CSV files in pretrain directory
    pretrain_files = glob.glob(os.path.join(args.pretrain_data_dir, "*.csv"))
    print(f"Found {len(pretrain_files)} pre-training files: {pretrain_files}")
    
    pretrained_models = []
    
    for file_path in pretrain_files:
        domain_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Determine frequency based on filename or data characteristics
        frequency = 'monthly'
        if 'daily' in domain_name.lower() or 'epidemiology' in domain_name.lower():
            frequency = 'daily'
        
        pretrainer = DomainPreTrainer(
            domain_name, VisionTransformerForTimeSeries, base_config, device
        )
        
        data = pretrainer.load_and_prepare_data(file_path, frequency=frequency)
        pretrained_state = pretrainer.pretrain(data, epochs=args.epochs_pretrain)
        
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
            cyber_data_loader = DataLoaderS(args.cyber_data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=6)
            
            # Prepare cyber data in the format needed for validation
            cyber_data = {
                'X_train': cyber_data_loader.train[0],
                'y_train': cyber_data_loader.train[1],
                'X_val': cyber_data_loader.valid[0],
                'y_val': cyber_data_loader.valid[1]
            }
            
            # Cyber threat model config
            cyber_config = base_config.copy()
            cyber_config['num_nodes'] = cyber_data_loader.m
            cyber_config['forecast_horizon'] = cyber_data_loader.valid_horizon
            
            # Run validation
            results = validate_transfer_learning_benefit(
                cyber_data, cyber_config, device, pretrained_models, epochs=30
            )
            
            # Save validation results
            with open('transfer_learning_pretrain/transfer_validation.json', 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            print(f"Error during transfer learning validation: {e}")
    
    # TRANSFER + INITIAL FINE-TUNING
    if pretrained_models:
        print("\n" + "="*80)
        print("PHASE 2: TRANSFER + INITIAL FINE-TUNING")
        print("="*80)
        
        # Load cyber threat data with operational horizon
        print("Loading cyber threat data for fine-tuning...")
        cyber_data_loader = DataLoaderS(args.cyber_data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=12)
        
        # Cyber threat model config
        cyber_config = base_config.copy()
        cyber_config['num_nodes'] = cyber_data_loader.m
        cyber_config['forecast_horizon'] = cyber_data_loader.valid_horizon
        
        # Transfer weights
        transfer_manager = TransferManager(
            pretrained_models, cyber_config, VisionTransformerForTimeSeries, device
        )
        
        transferred_model, transferred_layers = transfer_manager.create_transferred_model(
            strategy='selective'
        )
        
        # Initial fine-tuning
        print(f"\nInitial fine-tuning for {args.epochs_finetune} epochs...")
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(transferred_model.parameters(), lr=0.0005, weight_decay=0.01)  # Lower LR for fine-tuning
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(1, args.epochs_finetune + 1):
            # Training
            transferred_model.train()
            train_loss = 0
            n_batches = 0
            
            indices = torch.randperm(len(cyber_data_loader.train[0]))
            batch_size = cyber_config['batch_size']
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = cyber_data_loader.train[0][batch_indices].to(device)
                y_batch = cyber_data_loader.train[1][batch_indices].to(device)
                
                optimizer.zero_grad()
                output = transferred_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else train_loss
            
            # Validation
            transferred_model.eval()
            with torch.no_grad():
                val_output = transferred_model(cyber_data_loader.valid[0].to(device))
                val_loss = criterion(val_output, cyber_data_loader.valid[1].to(device))
            
            print(f"Fine-tune Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={val_loss.item():.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = transferred_model.state_dict().copy()
        
        # Save the fine-tuned model for hyperparameter optimization
        torch.save({
            'model_state_dict': best_state,
            'config': cyber_config,
            'transferred_layers': transferred_layers,
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'fine_tune_val_loss': best_val_loss
        }, 'transfer_learning_pretrain/transferred_model_for_hp_search.pt')
        
        print(f"\nPhase 2 completed. Fine-tuned model saved for hyperparameter optimization.")
        print(f"Final validation loss: {best_val_loss:.6f}")
    
    print(f"\nPre-training and transfer learning pipeline completed.")
    print(f"Results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()