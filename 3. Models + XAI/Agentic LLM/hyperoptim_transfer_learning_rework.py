
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

# TRANSFER LEARNING COMPONENTS 
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

def load_pretrained_models(pretrain_dir, device):
    """Load pre-trained models from transfer learning phase"""
    print(f"Loading pre-trained models from: {pretrain_dir}")
    
    pretrained_models = []
    pretrain_files = glob.glob(os.path.join(pretrain_dir, "pretrained_*.pt"))
    
    for file_path in pretrain_files:
        try:
            domain_name = os.path.splitext(os.path.basename(file_path))[0].replace('pretrained_', '')
            state_dict = torch.load(file_path, map_location=device)
            
            pretrained_models.append({
                'domain': domain_name,
                'state_dict': state_dict
            })
            print(f"  Loaded pre-trained model: {domain_name}")
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    print(f"Successfully loaded {len(pretrained_models)} pre-trained models")
    return pretrained_models

# MODEL ARCHITECTURE 

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

class PatchEmbedding(nn.Module):
    """Convert time series into patches with proper size handling """
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Ensure sequence length is divisible by patch size AND patch_size <= seq_len
        if seq_len % patch_size != 0 or patch_size > seq_len:
            # Find compatible patch sizes that divide sequence length AND are <= seq_len
            divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0 and i <= seq_len]
            if divisors:
                patch_size = max(divisors)  # Use largest compatible divisor
                print(f"Adjusted patch_size to: {patch_size} (compatible with seq_len {seq_len})")
            else:
                patch_size = 1  # Fallback
                print(f"Using fallback patch_size: 1")
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        # Additional safety check
        if self.num_patches <= 0:
            self.num_patches = 1
            self.patch_size = seq_len
            print(f"Emergency adjustment: patch_size={seq_len}, num_patches=1")
        
        print(f"Final: seq_len={seq_len}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Proper 1D convolution for time series
        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=self.patch_size,  # Use the adjusted patch_size
            stride=self.patch_size
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
        if config['sequence_length'] % config['patch_size'] != 0 or config['patch_size'] > config['sequence_length']:
            config['patch_size'] = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
            print(f"Validated patch_size: {config['patch_size']}")
            
        if config['embed_dim'] % config['num_heads'] != 0:
            config['num_heads'] = self._get_compatible_heads(config['embed_dim'], config['num_heads'])
            print(f"Validated num_heads: {config['num_heads']}")

    def _get_compatible_patch_size(self, seq_len, desired_patch_size):
        """Find compatible patch size that divides sequence length AND is <= seq_len"""
        divisors = []
        for i in range(1, seq_len + 1):
            if seq_len % i == 0 and i <= seq_len:  # Added i <= seq_len check
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


# DATA LOADER AND OTHER COMPONENTS 
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
        
        # Use shorter forecast horizon for validation and test if needed
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

# HYPERPARAMETER OPTIMIZATION WITH PRE-TRAINED INITIALIZATION

def initialize_model_with_transfer(hp, pretrained_models, device, transfer_strategy='selective'):
    """Initialize model with pre-trained weights for hyperparameter optimization"""
    print("Initializing model with pre-trained weights...")
    
    # Create base model
    model = VisionTransformerForTimeSeries(hp).to(device)
    
    if pretrained_models:
        # Transfer pre-trained weights
        transfer_manager = TransferManager(pretrained_models, hp, VisionTransformerForTimeSeries, device)
        model, transferred_layers = transfer_manager.create_transferred_model(strategy=transfer_strategy)
        print(f"Successfully transferred {len(transferred_layers)} layers")
    else:
        print("No pre-trained models available, using random initialization")
    
    return model

def create_directories():
    """Create all necessary directories for plots and models"""
    directories = [
        'model/ViT/Validation',
        'model/ViT/Testing', 
        'model/ViT/plots',
        'model/ViT/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def consistent_name(name):
    """Standardise names for plotting and saving."""
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
    """Compute and save metrics for a single node"""
    # Ensure we have valid tensors
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
    with open(f'model/ViT/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')
        f.close()

def plot_predicted_actual(predicted, actual, title, type, confidence_95=None):
    """Plot predicted vs actual curves with confidence intervals - for proper time range"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    M = []
    
    # Create proper time range from 2011 to 2019
    for year in range(11, 20):  # 2011 to 2019
        for month in months:
            # Skip months before July 2011 if needed
            if year == 11 and month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']:
                continue
            M.append(month + '-' + str(year))
    
    # Extend to 2020 if needed for longer sequences
    if len(predicted) > len(M):
        for year in range(20, 25):  # 2020 to 2024 if needed
            for month in months:
                M.append(month + '-' + str(year))
    
    M2 = []
    p = []
    
    # For both Validation and Testing, use appropriate time range
    if type == 'Testing':
        # Testing should use recent data
        M = M[-len(predicted):] if len(predicted) <= len(M) else M
    else:
        # Validation should cover 2011-2019 period
        # Adjust to ensure we have enough months for the plot
        if len(predicted) <= len(M):
            M = M[:len(predicted)]
        else:
            # Extend if prediction is longer than available labels
            current_len = len(M)
            for i in range(len(predicted) - current_len):
                M.append(f"Month_{i+1}")
    
    # Select every 3rd month for x-axis labels to avoid overcrowding
    for index, value in enumerate(M):
        if index % 3 == 0:  # Show every 3rd month
            M2.append(value)
            p.append(index + 1)
    
    # If we have too few labels, add more
    if len(M2) < 4 and len(M) > 0:
        M2 = [M[i] for i in range(0, len(M), max(1, len(M)//4))]
        p = [i+1 for i in range(0, len(M), max(1, len(M)//4))]

    x = range(1, len(predicted) + 1)
    plt.figure(figsize=(14, 6))
    plt.plot(x, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted, '--', color='purple', label='Predicted', linewidth=2)
    
    # Add confidence intervals if provided
    if confidence_95 is not None and len(confidence_95) == len(predicted):
        plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, 
                        alpha=0.3, color='pink', label='95% Confidence')
        print(f"  Added confidence intervals to plot")
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(f"{title} - {type}", y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Time", fontsize=15)
    
    if p and M2:  # Only set ticks if we have valid labels
        plt.xticks(ticks=p, labels=M2, rotation=45, fontsize=11)
    else:
        plt.xticks(fontsize=11)
        
    plt.yticks(fontsize=11)
    
    title = title.replace('/', '_')
    plt.savefig(f'model/ViT/{type}/{title}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/ViT/{type}/{title}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()
    print(f"  Saved plot: model/ViT/{type}/{title}_{type}.png")

def calculate_rrse_rae_comprehensive(predict, test, scale):
    """Comprehensive RRSE and RAE calculation following the original paper's methodology"""
    if predict.numel() == 0 or test.numel() == 0:
        print("Warning: Empty tensors in metric calculation")
        return float('inf'), float('inf')
    
    try:
        # Denormalize the data
        predict_denorm = predict * scale
        test_denorm = test * scale
        
        # Flatten all dimensions for overall metrics
        predict_flat = predict_denorm.flatten()
        test_flat = test_denorm.flatten()
        
        # Calculate numerator for RRSE (Root Relative Squared Error)
        squared_errors = (test_flat - predict_flat) ** 2
        sum_squared_errors = torch.sum(squared_errors)
        root_sum_squared_errors = torch.sqrt(sum_squared_errors)
        
        # Calculate denominator for RRSE
        # This is the root of sum of squared differences from mean
        test_mean = torch.mean(test_flat)
        squared_deviations = (test_flat - test_mean) ** 2
        sum_squared_deviations = torch.sum(squared_deviations)
        root_sum_squared_deviations = torch.sqrt(sum_squared_deviations)
        
        # Calculate RRSE
        if root_sum_squared_deviations > 0:
            rrse = root_sum_squared_errors / root_sum_squared_deviations
        else:
            rrse = float('inf')
        
        # Calculate numerator for RAE (Relative Absolute Error)
        absolute_errors = torch.abs(test_flat - predict_flat)
        sum_absolute_errors = torch.sum(absolute_errors)
        
        # Calculate denominator for RAE
        absolute_deviations = torch.abs(test_flat - test_mean)
        sum_absolute_deviations = torch.sum(absolute_deviations)
        
        # Calculate RAE
        if sum_absolute_deviations > 0:
            rae = sum_absolute_errors / sum_absolute_deviations
        else:
            rae = float('inf')
        
        # Convert to Python floats
        rrse = rrse.item() if isinstance(rrse, torch.Tensor) else rrse
        rae = rae.item() if isinstance(rae, torch.Tensor) else rae
        
        return rrse, rae
        
    except Exception as e:
        print(f"Error in comprehensive metric calculation: {e}")
        return float('inf'), float('inf')

def evaluate_comprehensive(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot, device, eval_type="Validation"):
    """Comprehensive evaluation function with proper metric calculation and plotting - """
    model.train()  # Keep dropout active for Bayesian estimation
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    all_predictions = []
    all_targets = []
    all_variances = []  # Store variances for confidence intervals

    if X.size(0) == 0:
        print(f"CRITICAL: No {eval_type} data available for evaluation")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        return float('inf'), float('inf')

    print(f"Evaluating on {X.size(0)} {eval_type.lower()} sequences...")

    batch_count = 0
    for batch_idx, (X_batch, Y_batch) in enumerate(data.get_batches(X, Y, batch_size, False)):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Bayesian estimation with multiple runs for confidence intervals
        num_runs = 10  # Increased for better confidence intervals
        outputs = []
        with torch.no_grad():
            for run in range(num_runs):
                output = model(X_batch)
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean_prediction = torch.mean(outputs, dim=0)
        
        # Ensure prediction horizon matches target horizon
        min_horizon = min(mean_prediction.shape[1], Y_batch.shape[1])
        mean_prediction = mean_prediction[:, :min_horizon, :]
        variance = torch.var(outputs[:, :, :min_horizon, :], dim=0)  # Calculate variance for confidence intervals
        std_dev = torch.sqrt(variance)  # Standard deviation
        confidence_95 = 1.96 * std_dev / math.sqrt(num_runs)  # 95% confidence interval

        # Store predictions, targets, and confidence intervals
        scale_batch = data.scale.expand(mean_prediction.size(0), mean_prediction.size(1), data.m).to(device)
        
        # Denormalize for metric calculation
        predictions_denorm = mean_prediction * scale_batch
        targets_denorm = Y_batch[:, :min_horizon, :] * scale_batch  # Match target horizon
        confidence_denorm = confidence_95 * scale_batch

        all_predictions.append(predictions_denorm.cpu())
        all_targets.append(targets_denorm.cpu())
        all_variances.append(confidence_denorm.cpu())  # Store confidence intervals

        # Calculate losses (still using normalized data for consistency)
        output_normalized = mean_prediction
        Y_normalized = Y_batch[:, :min_horizon, :]  # Match target horizon
        
        total_loss += evaluateL2(output_normalized, Y_normalized).item()
        total_loss_l1 += evaluateL1(output_normalized, Y_normalized).item()
        n_samples += (output_normalized.size(0) * output_normalized.size(1) * data.m)
        batch_count += 1

    if not all_predictions:
        print(f"CRITICAL: No predictions were generated during {eval_type.lower()} evaluation")
        return float('inf'), float('inf')

    # Concatenate all predictions, targets, and confidence intervals
    try:
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        all_confidence_tensor = torch.cat(all_variances, dim=0)  # Concatenate confidence intervals
        
        print(f"{eval_type} evaluation completed: {batch_count} batches, {all_predictions_tensor.shape} predictions")
        
        # Calculate comprehensive metrics
        rrse, rae = calculate_rrse_rae_comprehensive(all_predictions_tensor, all_targets_tensor, data.scale)
        
        print(f"Calculated {eval_type.lower()} metrics - RRSE: {rrse:.6f}, RAE: {rae:.6f}")
        
        # Plot results if requested
        if is_plot and all_predictions_tensor.numel() > 0:
            plot_evaluation_results(all_predictions_tensor, all_targets_tensor, all_confidence_tensor, data, eval_type)
            
        return rrse, rae
        
    except Exception as e:
        print(f"Error in {eval_type.lower()} evaluation aggregation: {e}")
        return float('inf'), float('inf')

def plot_evaluation_results(predictions, targets, confidence, data, eval_type): 
    """Plot evaluation results for sample nodes with confidence intervals"""
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    confidence_np = confidence.numpy()
    
    # Plot first few nodes (columns)
    num_nodes_to_plot = min(645, data.m)
    print(f"Creating {num_nodes_to_plot} {eval_type.lower()} plots with confidence intervals...")

    for col in range(num_nodes_to_plot):
        # Use the real column header for the node name
        if hasattr(data, "feature_names") and len(data.feature_names) > col:
            node_name = str(data.feature_names[col])
        elif hasattr(data, "col") and len(data.col) > col:
            node_name = str(data.col[col])
        else:
            node_name = f"Node_{col}"
        
        node_name = consistent_name(node_name)

        # Use the last sequence for plotting
        if predictions_np.shape[0] > 0:
            pred_curve = predictions_np[-1, :, col]
            target_curve = targets_np[-1, :, col]
            confidence_curve = confidence_np[-1, :, col]

            # Save metrics and plots with descriptive names
            save_metrics_1d(torch.from_numpy(pred_curve), torch.from_numpy(target_curve), node_name, eval_type)
            plot_predicted_actual(pred_curve, target_curve, node_name, eval_type, confidence_curve)


def train_stable(data, X, Y, model, criterion, optimizer, batch_size, device):
    """Stable training function with gradient monitoring """
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
        
        # Ensure output and Y_batch have the same forecast horizon
        min_horizon = min(output.shape[1], Y_batch.shape[1])
        output = output[:, :min_horizon, :]
        Y_batch = Y_batch[:, :min_horizon, :]
        
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

def sample_compatible_hyperparameters(search_space, num_nodes, data_out_len, operational_horizon=36):
    """Sample hyperparameters with better compatibility checks"""
    max_attempts = 100
    for attempt in range(max_attempts):
        hp = {}
        for key, values in search_space.items():
            hp[key] = random.choice(values)
        
        # Apply compatibility constraints
        # 1. embed_dim % num_heads == 0
        if hp['embed_dim'] % hp['num_heads'] != 0:
            compatible_heads = [h for h in search_space['num_heads'] if hp['embed_dim'] % h == 0]
            if compatible_heads:
                hp['num_heads'] = random.choice(compatible_heads)
            else:
                continue
        
        # 2. sequence_length % patch_size == 0 AND patch_size <= sequence_length
        sequence_length = hp['sequence_length']
        patch_size = hp['patch_size']
        
        # Check if patch_size is compatible
        if sequence_length % patch_size != 0 or patch_size > sequence_length:
            # Find compatible patch sizes
            compatible_patches = [p for p in search_space['patch_size'] 
                                if sequence_length % p == 0 and p <= sequence_length]
            if compatible_patches:
                hp['patch_size'] = random.choice(compatible_patches)
            else:
                # If no compatible patches, find divisors of sequence_length
                divisors = [i for i in range(1, sequence_length + 1) if sequence_length % i == 0]
                if divisors:
                    hp['patch_size'] = max(divisors)
                else:
                    hp['patch_size'] = 1  # Fallback
                print(f"  Adjusted patch_size to {hp['patch_size']} for sequence_length {sequence_length}")
        
        # 3. hidden_dim >= embed_dim
        if hp['hidden_dim'] < hp['embed_dim']:
            compatible_hidden = [h for h in search_space['hidden_dim'] if h >= hp['embed_dim']]
            if compatible_hidden:
                hp['hidden_dim'] = random.choice(compatible_hidden)
            else:
                continue
        
        # 4. Memory constraint
        memory_estimate = hp['batch_size'] * hp['sequence_length'] * hp['embed_dim'] * num_nodes
        if memory_estimate > 1e9:
            smaller_batches = [bs for bs in search_space['batch_size'] if bs < hp['batch_size']]
            if smaller_batches:
                hp['batch_size'] = max(smaller_batches)
            else:
                continue
        
        # 5. Use optimization horizon for search, but store operational horizon separately
        hp['optimization_horizon'] = data_out_len  # For current optimization
        hp['operational_horizon'] = operational_horizon  # For final model
        
        # Additional safety check: ensure patch_size is valid
        if hp['patch_size'] > hp['sequence_length']:
            hp['patch_size'] = min(hp['patch_size'], hp['sequence_length'])
        
        # All constraints satisfied
        return hp
    
    # Fallback with optimization horizon and safe parameters
    print("Warning: Could not find fully compatible hyperparameters, using safe fallback")
    fallback_hp = {
        'sequence_length': 12,
        'patch_size': 4,  # Safe patch size that divides 12
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
        'forecast_horizon': data_out_len,  # Optimization horizon
        'optimization_horizon': data_out_len,
        'operational_horizon': operational_horizon,
        'grad_clip_value': 1.0,
        'activation': 'relu'
    }
    return fallback_hp


def main():
    parser = argparse.ArgumentParser(description='ViT Hyperparameter Optimization with Transfer Learning')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--iterations', type=int, default=60, help='number of random search iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--L1Loss', type=bool, default=True)
    
    # NEW ARGUMENTS FOR TRANSFER LEARNING
    parser.add_argument('--pretrain_dir', type=str, default='./transfer_learning_pretrain/', 
                       help='directory with pre-trained models from transfer learning')
    parser.add_argument('--transfer_strategy', type=str, default='selective', 
                       choices=['single', 'ensemble', 'selective'],
                       help='strategy for transferring pre-trained weights')
    parser.add_argument('--fine_tune_all', action='store_true', default=True,
                       help='fine-tune all layers (if False, only fine-tune head)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create directories first
    create_directories()
    
    # Set fixed random seed
    set_random_seed(123)
    
    # Load pre-trained models
    pretrained_models = load_pretrained_models(args.pretrain_dir, device)
    
    # Use 12-month horizon for optimization, 36-month for final operational model
    OPTIMIZATION_HORIZON = 12  # For hyperparameter optimization
    OPERATIONAL_HORIZON = 36   # For final operational forecasting
    
    print("Loading data with optimized horizon configuration...")
    try:
        # Use shorter horizon for optimization to ensure we have validation data
        data = DataLoaderS(args.data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=OPTIMIZATION_HORIZON)
        print(f"Data loaded successfully!")
        print(f"Optimization horizon: {data.out_len} months")
        print(f"Operational horizon (for final model): {OPERATIONAL_HORIZON} months")
        print(f"Using transfer learning: {len(pretrained_models)} pre-trained models loaded")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Hyperparameter search space - use optimization horizon
    search_space = {
        # Architecture: Sequence and Patch Configuration
        'sequence_length': [8, 12, 16, 24],
        'patch_size': [1, 2, 4, 6, 8],
        
        # Architecture: Embedding Dimensions
        'embed_dim': [64, 128, 256],
        
        # Architecture: Attention Mechanism
        'num_heads': [4, 8],
        
        # Architecture: Feed-Forward Network
        'hidden_dim': [128, 256, 512],
        
        # Architecture: Depth
        'num_layers': [2, 3, 4],
        
        # Regularization: Dropout Rates
        'dropout': [0.1, 0.2, 0.3],
        'mc_dropout': [0.1, 0.2, 0.3],
        
        # Optimization: Learning Configuration
        'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],
        'weight_decay': [0.0, 0.001, 0.01],
        'batch_size': [4, 8, 16],
        
        # Optimization: Learning Rate Scheduling
        'scheduler_patience': [5, 7, 10],
        'scheduler_factor': [0.5, 0.7],
        
        # Architecture: Positional Encoding
        'positional_encoding_type': ['learned', 'sinusoidal'],
        
        # Monte Carlo: Bayesian Inference
        'mc_inference_runs': [10, 20, 30],
        
        # Forecast Configuration - use optimization horizon
        'forecast_horizon': [OPTIMIZATION_HORIZON],
        
        # Training: Gradient Management
        'grad_clip_value': [1.0, 2.0],
        
        # Architecture: Activation Functions
        'activation': ['relu', 'gelu'],
    }
    
    print("Starting hyperparameter optimization with transfer learning...")
    print(f"Training sequences: {data.train[0].shape[0]}")
    print(f"Validation sequences: {data.valid[0].shape[0]}")
    print(f"Test sequences: {data.test[0].shape[0]}")
    print(f"Number of nodes: {data.m}")
    print(f"Optimization horizon: {OPTIMIZATION_HORIZON} months")
    print(f"Final operational horizon: {OPERATIONAL_HORIZON} months")
    print(f"Transfer strategy: {args.transfer_strategy}")
    print(f"Fine-tune all layers: {args.fine_tune_all}")

    best_val_rrse = float('inf')
    best_val_rae = float('inf')
    best_hp = None
    best_model_state = None
    best_loss = float('inf')
    best_epoch = 0
    
    successful_iterations = 0
    all_results = []
    
    for iteration in range(args.iterations):
        print(f"\n Iteration {iteration + 1}/{args.iterations} ")
        
        # Sample compatible hyperparameters - pass data.out_len to ensure matching forecast horizon
        hp = sample_compatible_hyperparameters(search_space, data.m, data.out_len, OPERATIONAL_HORIZON)
        hp['num_nodes'] = data.m  
        
        print(f"Hyperparameters: {hp}")
        
        try:
            # Create model WITH PRE-TRAINED INITIALIZATION
            print("Initializing model with pre-trained weights...")
            model = initialize_model_with_transfer(hp, pretrained_models, device, args.transfer_strategy)
            
            # Test a forward pass with a small batch to catch configuration errors early
            print("Testing model configuration with forward pass...")
            with torch.no_grad():
                test_batch = data.train[0][:2].to(device)  # Use first 2 samples
                test_output = model(test_batch)
                print(f"  Forward pass successful: input {test_batch.shape} -> output {test_output.shape}")
            
            # If not fine-tuning all layers, freeze transferred layers
            if not args.fine_tune_all and pretrained_models:
                print("Freezing transferred layers, only training forecast head...")
                for name, param in model.named_parameters():
                    if any(layer in name for layer in ['transformer', 'patch_embed', 'position_embeddings']):
                        param.requires_grad = False
            
            # Use stable training configuration
            criterion = nn.MSELoss()
            evaluateL2 = nn.MSELoss()
            evaluateL1 = nn.L1Loss()
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=hp['learning_rate'], 
                weight_decay=hp['weight_decay']
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hp['scheduler_patience'], factor=hp['scheduler_factor'])
            
            # Training loop
            best_iter_val_rrse = float('inf')
            best_iter_val_rae = float('inf')
            best_iter_loss = float('inf')
            patience_counter = 0
            best_model_state_iter = None
            
            print("Starting training...")
            for epoch in range(1, args.epochs + 1):
                # Training phase
                train_loss = train_stable(data, data.train[0], data.train[1], model, criterion, optimizer, hp['batch_size'], device)
                
                # Validation phase
                val_rrse, val_rae = evaluate_comprehensive(data, data.valid[0], data.valid[1], model, evaluateL2, evaluateL1, hp['batch_size'], False, device, "Validation")
                
                scheduler.step(val_rrse if not math.isinf(val_rrse) else train_loss)
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val RRSE: {val_rrse:.6f}, Val RAE: {val_rae:.6f}')
                
                # Save best model for this iteration
                if (not math.isinf(val_rrse) and not math.isnan(val_rrse) and 
                    val_rrse < best_iter_val_rrse):
                    best_iter_val_rrse = val_rrse
                    best_iter_val_rae = val_rae
                    best_iter_loss = train_loss
                    best_model_state_iter = model.state_dict().copy()
                    patience_counter = 0
                    best_epoch = epoch
                    print(f"  -> New best for iteration: RRSE={val_rrse:.6f}, RAE={val_rae:.6f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= 15:  # Increased patience for longer training
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Store results for this iteration
            iteration_result = {
                'iteration': iteration + 1,
                'hyperparameters': hp,
                'best_train_loss': best_iter_loss,
                'best_val_rrse': best_iter_val_rrse,
                'best_val_rae': best_iter_val_rae,
                'epochs_trained': epoch,
                'transfer_strategy': args.transfer_strategy,
                'fine_tune_all': args.fine_tune_all,
                'pretrained_models_used': len(pretrained_models)
            }
            all_results.append(iteration_result)
            
            # Update global best
            if (not math.isinf(best_iter_val_rrse) and not math.isnan(best_iter_val_rrse) and 
                best_iter_val_rrse < best_val_rrse and best_model_state_iter is not None):
                best_val_rrse = best_iter_val_rrse
                best_val_rae = best_iter_val_rae
                best_loss = best_iter_loss
                best_hp = hp.copy()
                best_model_state = best_model_state_iter
                successful_iterations += 1
                print(f"*** NEW GLOBAL BEST! Iteration {iteration + 1} - RRSE: {best_val_rrse:.6f}, RAE: {best_val_rae:.6f} ***")
            else:
                print(f"Iteration {iteration + 1} completed - Best RRSE: {best_iter_val_rrse:.6f}")
                
        except Exception as e:
            print(f"Iteration {iteration + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final evaluation and plotting with best model
    print("\n FINAL EVALUATION AND PLOTTING ")
    if best_hp is not None and best_model_state is not None:
        # Save best hyperparameters with transfer learning info
        best_hp['transfer_strategy'] = args.transfer_strategy
        best_hp['fine_tune_all'] = args.fine_tune_all
        best_hp['pretrained_models_used'] = len(pretrained_models)
        
        with open('model/ViT/hp.txt', 'w') as f:
            json.dump(best_hp, f, indent=2, cls=NumpyEncoder)
        
        # Load best model for final evaluation
        model = VisionTransformerForTimeSeries(best_hp).to(device)
        model.load_state_dict(best_model_state)
        
        # Final validation evaluation with plots and confidence intervals
        print("Final validation evaluation with confidence intervals...")
        val_rrse, val_rae = evaluate_comprehensive(data, data.valid[0], data.valid[1], model, 
                                                 nn.MSELoss(), nn.L1Loss(), best_hp['batch_size'], True, device, "Validation")
        
        # Final test evaluation with plots and confidence intervals
        if data.test[0].size(0) > 0:
            print("Final test evaluation with confidence intervals...")
            test_rrse, test_rae = evaluate_comprehensive(data, data.test[0], data.test[1], model, 
                                                       nn.MSELoss(), nn.L1Loss(), best_hp['batch_size'], True, device, "Testing")
        else:
            print("No test data available for evaluation")
            test_rrse, test_rae = float('inf'), float('inf')
        
        # Save the best model with transfer learning metadata
        model_save_data = {
            'model_state_dict': best_model_state,
            'hyperparameters': best_hp,
            'validation_rrse': val_rrse,
            'validation_rae': val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'successful_iterations': successful_iterations,
            'timestamp': datetime.now().isoformat(),
            'transfer_learning_info': {
                'strategy': args.transfer_strategy,
                'fine_tune_all': args.fine_tune_all,
                'pretrained_models_count': len(pretrained_models),
                'pretrained_domains': [p['domain'] for p in pretrained_models] if pretrained_models else []
            }
        }
        
        torch.save(model_save_data, 'model/ViT/best_hyperparameter_model.pt')
        
        # Save training configuration and results with transfer learning info
        training_summary = {
            'hyperparameters': best_hp,
            'final_loss': best_loss,
            'total_epochs': best_epoch,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.raw_data.shape,
            'device': str(device),
            'best_val_rrse': best_val_rrse,
            'best_val_rae': best_val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'successful_iterations': successful_iterations,
            'total_iterations': args.iterations,
            'transfer_learning': {
                'enabled': True,
                'strategy': args.transfer_strategy,
                'fine_tune_all': args.fine_tune_all,
                'pretrained_models_loaded': len(pretrained_models)
            }
        }
        
        with open('model/ViT/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
        print("Training summary saved as 'model/ViT/training_summary.json'")
        
        print(f"\n TRANSFER LEARNING HYPERPARAMETER OPTIMIZATION COMPLETE ")
        print(f"Best model saved as 'model/ViT/best_hyperparameter_model.pt'")
        print(f"Final Validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if data.test[0].size(0) > 0:
            print(f"Final Testing - RRSE: {test_rrse:.6f}, RAE: {test_rae:.6f}")
        print(f"Transfer learning: {len(pretrained_models)} pre-trained models used")
        print(f"Transfer strategy: {args.transfer_strategy}")
        print(f"Fine-tune all layers: {args.fine_tune_all}")
        
    else:
    # Fallback: create and save a model with default parameters
        print("\n CREATING FALLBACK MODEL ")
        default_hp = {
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
            'forecast_horizon': data.out_len,  # Use data's out_len
            'grad_clip_value': 1.0,
            'activation': 'relu',
            'num_nodes': data.m,
            # Add transfer learning info to default hyperparameters
            'transfer_strategy': args.transfer_strategy,
            'fine_tune_all': args.fine_tune_all,
            'pretrained_models_used': len(pretrained_models)
        }
        
        with open('model/ViT/hp.txt', 'w') as f:
            json.dump(default_hp, f, indent=2, cls=NumpyEncoder)
        
        # Try to initialize with pre-trained weights if available, otherwise use random
        if pretrained_models:
            print("Attempting to create fallback model with pre-trained weights...")
            try:
                model = initialize_model_with_transfer(default_hp, pretrained_models, device, args.transfer_strategy)
                print("Fallback model created with pre-trained weights")
            except Exception as e:
                print(f"Failed to create model with pre-trained weights: {e}")
                print("Creating fallback model with random initialization...")
                model = VisionTransformerForTimeSeries(default_hp).to(device)
        else:
            print("Creating fallback model with random initialization...")
            model = VisionTransformerForTimeSeries(default_hp).to(device)
        
        # If not fine-tuning all layers and we have pre-trained models, freeze transferred layers
        if not args.fine_tune_all and pretrained_models:
            print("Freezing transferred layers in fallback model...")
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['transformer', 'patch_embed', 'position_embeddings']):
                    param.requires_grad = False
        
        # Create some basic plots even with fallback model
        val_rrse, val_rae = float('inf'), float('inf')
        if data.valid[0].size(0) > 0:
            try:
                val_rrse, val_rae = evaluate_comprehensive(data, data.valid[0], data.valid[1], model, 
                                    nn.MSELoss(), nn.L1Loss(), default_hp['batch_size'], True, device, "Validation")
                print("Fallback model evaluated on validation data and plots created.")
            except Exception as e:
                print(f"Error evaluating fallback model: {e}")
                val_rrse, val_rae = float('inf'), float('inf')
        
        # Save fallback training summary with transfer learning info
        training_summary = {
            'hyperparameters': default_hp,
            'final_loss': float('inf'),
            'total_epochs': 0,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.raw_data.shape,
            'device': str(device),
            'best_val_rrse': val_rrse,
            'best_val_rae': val_rae,
            'test_rrse': float('inf'),
            'test_rae': float('inf'),
            'successful_iterations': 0,
            'total_iterations': args.iterations,
            'is_fallback': True,
            'transfer_learning': {
                'enabled': len(pretrained_models) > 0,
                'strategy': args.transfer_strategy,
                'fine_tune_all': args.fine_tune_all,
                'pretrained_models_loaded': len(pretrained_models),
                'fallback_initialization_successful': len(pretrained_models) > 0
            },
            'fallback_reason': 'No successful hyperparameter optimization iterations'
        }
        
        with open('model/ViT/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
        
        # Save fallback model with transfer learning metadata
        model_save_data = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': default_hp,
            'is_fallback': True,
            'timestamp': datetime.now().isoformat(),
            'transfer_learning_info': {
                'strategy': args.transfer_strategy,
                'fine_tune_all': args.fine_tune_all,
                'pretrained_models_count': len(pretrained_models),
                'pretrained_domains': [p['domain'] for p in pretrained_models] if pretrained_models else [],
                'fallback_used': True,
                'fallback_reason': 'No successful optimization iterations'
            },
            'validation_rrse': val_rrse,
            'validation_rae': val_rae
        }
        
        torch.save(model_save_data, 'model/ViT/best_hyperparameter_model.pt')
        
        print("Fallback model saved as 'model/ViT/best_hyperparameter_model.pt'")
        print(f"Fallback model validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if pretrained_models:
            print(f"Fallback model used {len(pretrained_models)} pre-trained models")
            print(f"Transfer strategy: {args.transfer_strategy}")
            print(f"Fine-tune all layers: {args.fine_tune_all}")
        else:
            print("Fallback model used random initialization (no pre-trained models available)")
    
    print(f"\nHyperparameter optimization with transfer learning completed.")
    print(f"Plots with confidence intervals saved in: model/ViT/Validation/ and model/ViT/Testing/")
    print(f"Model saved as: model/ViT/best_hyperparameter_model.pt")
    print(f"Hyperparameters saved as: model/ViT/hp.txt")
    print(f"Training summary saved as: model/ViT/training_summary.json")



if __name__ == "__main__":
    main()