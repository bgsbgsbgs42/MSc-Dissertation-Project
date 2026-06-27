
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
from sklearn.preprocessing import StandardScaler

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

class DataLoaderEnsemble:
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

def create_directories():
    """Create all necessary directories for plots and models"""
    directories = [
        'model/Ensemble/Validation',
        'model/Ensemble/Testing', 
        'model/Ensemble/plots',
        'model/Ensemble/data'
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
    with open(f'model/Ensemble/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')
        f.close()

def plot_predicted_actual(predicted, actual, title, type, confidence_95=None):
    """Plot predicted vs actual curves with confidence intervals """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    M = []
    
    # Generate proper timeline from July 2011 to December 2024
    for year in range(2011, 2025):   
        for month in months:
            if year == 2011 and month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']:
                continue
            if year == 2024 and month in ['Nov', 'Dec']:
                # Only include if we have data for these months
                continue
            M.append(f"{month}-{str(year)[-2:]}")   
    
    M2 = []
    p = []
    
    # Proper timeline selection for validation and testing
    if type == 'Testing':
        # Testing: Last 36 months (3 years)
        M = M[-len(predicted):] if len(predicted) <= len(M) else M
        for index, value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index + 1) 
    else:
        # Validation: October 2016 to September 2019 (36 months)
        # Find the start index for Oct-16
        start_idx = M.index('Oct-16') if 'Oct-16' in M else 63
        end_idx = start_idx + len(predicted)
        validation_months = M[start_idx:end_idx]
        
        for index, value in enumerate(validation_months):
            if 'Dec' in validation_months[index] or 'Mar' in validation_months[index] or 'Jun' in validation_months[index] or 'Sep' in validation_months[index]:
                M2.append(validation_months[index])
                p.append(index + 1) 

    x = range(1, len(predicted) + 1)
    plt.figure(figsize=(12, 6))
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
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13) 
    plt.yticks(fontsize=13)
    
    title = title.replace('/', '_')
    plt.savefig(f'model/Ensemble/{type}/{title}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/Ensemble/{type}/{title}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()
    print(f"  Saved plot with confidence intervals: model/Ensemble/{type}/{title}_{type}.png")

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
    """Comprehensive evaluation function with proper metric calculation and plotting"""
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

        #Ensure the model's forecast horizon matches the data
        model_forecast_horizon = model.config['forecast_horizon']
        data_forecast_horizon = Y_batch.shape[1]
        
        if model_forecast_horizon != data_forecast_horizon:
            print(f"Warning: Model forecast horizon ({model_forecast_horizon}) != Data forecast horizon ({data_forecast_horizon})")
            # Use the minimum of both
            actual_forecast_horizon = min(model_forecast_horizon, data_forecast_horizon)
        else:
            actual_forecast_horizon = model_forecast_horizon

        # Bayesian estimation with multiple runs for confidence intervals
        num_runs = 10
        outputs = []

        with torch.no_grad():
            for run in range(num_runs):
                output = model(X_batch)
                # Ensure output matches target dimensions
                if output.shape[1] != actual_forecast_horizon:
                    output = output[:, :actual_forecast_horizon, :]
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean_prediction = torch.mean(outputs, dim=0)
        variance = torch.var(outputs, dim=0)
        std_dev = torch.sqrt(variance)
        confidence_95 = 1.96 * std_dev / math.sqrt(num_runs)

        #Ensure scale batch matches prediction dimensions
        scale_batch = data.scale.expand(mean_prediction.size(0), mean_prediction.size(1), data.m).to(device)
        
        # Denormalize for metric calculation
        predictions_denorm = mean_prediction * scale_batch
        targets_denorm = Y_batch[:, :actual_forecast_horizon, :] * scale_batch
        confidence_denorm = confidence_95 * scale_batch

        all_predictions.append(predictions_denorm.cpu())
        all_targets.append(targets_denorm.cpu())
        all_variances.append(confidence_denorm.cpu())

        # Calculate losses (using normalized data for consistency)
        output_normalized = mean_prediction
        Y_normalized = Y_batch[:, :actual_forecast_horizon, :]
        
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
        all_confidence_tensor = torch.cat(all_variances, dim=0)
        
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

        # Use the last sequence for plotting (most recent period)
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

# Hyperparameter Search Space 
search_space = {
    # Temporal Parameters
    'sequence_length': [12, 24, 36],  # Input window size
    'forecast_horizon': [36],  #  36 months ahead
    
    # Vision Transformer Branch 
    # Patch embedding configuration
    'vit_patch_size': [2, 3, 4, 6],  # Must be divisors of sequence_length
    'vit_embed_dim': [64, 96, 128, 160],  # Embedding dimension
    'vit_num_heads': [4, 8],  # Attention heads (must divide embed_dim)
    'vit_hidden_dim': [128, 192, 256, 320],  # FFN hidden dimension
    'vit_num_layers': [2, 3, 4],  # Transformer encoder depth
    
    # Graph-Aware Branch 
    'graph_hidden_dim': [64, 96, 128, 160],  # Node representation dimension
    'graph_num_layers': [1, 2, 3],  # Graph encoder depth
    
    # Fusion Mechanism 
    'fusion_dim': [128, 192, 256, 320],  # Fusion head hidden dimension
    
    # Regularization 
    'dropout': [0.1, 0.2, 0.3],  # Standard dropout rate
    'mc_dropout': [0.1, 0.2, 0.3, 0.4],  # Monte Carlo dropout for uncertainty
    
    #  Optimization 
    'learning_rate': [0.0001, 0.0003, 0.0005, 0.001, 0.002],  # AdamW learning rate
    'weight_decay': [0.001, 0.01, 0.05],  # L2 regularization
    'batch_size': [4, 8, 16],  # Training batch size
    
    # Advanced Optimization 
    'optimizer_type': ['adamw', 'adam'],  # Optimizer variant
    'scheduler_patience': [5, 10, 15],  # LR scheduler patience
    'scheduler_factor': [0.3, 0.5, 0.7],  # LR reduction factor
    'gradient_clip': [0.5, 1.0, 2.0],  # Gradient clipping threshold
    
    # Ensemble Configuration 
    'num_mc_samples': [10, 15, 20],  # MC dropout samples for uncertainty
    'ensemble_weight_decay': [0.0, 0.001, 0.01],  # Weight averaging regularization
}

#  Validation Constraints 
validation_rules = {
    # Ensure patch_size divides sequence_length
    'patch_compatibility': lambda hp: hp['sequence_length'] % hp['vit_patch_size'] == 0,
    
    # Ensure num_heads divides embed_dim
    'attention_compatibility': lambda hp: hp['vit_embed_dim'] % hp['vit_num_heads'] == 0,
    
    # Ensure reasonable model capacity
    'capacity_constraint': lambda hp: (
        hp['vit_embed_dim'] * hp['vit_num_layers'] + 
        hp['graph_hidden_dim'] * hp['graph_num_layers']
    ) < 2000,  # Prevent excessive memory usage
    
    # Ensure fusion dimension is sufficient
    'fusion_constraint': lambda hp: hp['fusion_dim'] >= max(
        hp['vit_embed_dim'], hp['graph_hidden_dim']
    ),
}

# Sampling Strategy 
def sample_hyperparameters(search_space, validation_rules, data, max_attempts=100):
    """Sample valid hyperparameters with constraint checking"""
    for attempt in range(max_attempts):
        hp = {key: random.choice(values) for key, values in search_space.items()}
        
        # Fixed parameters
        hp['num_nodes'] = data.m  # Set from data
        
        # Apply validation rules
        if all(rule(hp) for rule in validation_rules.values()):
            return hp
    
    # Fallback to conservative configuration
    return get_conservative_config(data.m)

def get_conservative_config(num_nodes):
    """Fallback configuration guaranteed to work"""
    return {
        'sequence_length': 12,
        'forecast_horizon': 36,
        'vit_patch_size': 4,  # 12 % 4 = 0
        'vit_embed_dim': 128,  # Divisible by 4 and 8
        'vit_num_heads': 8,
        'vit_hidden_dim': 256,
        'vit_num_layers': 3,
        'graph_hidden_dim': 128,
        'graph_num_layers': 2,
        'fusion_dim': 256,
        'dropout': 0.2,
        'mc_dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 8,
        'optimizer_type': 'adamw',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'gradient_clip': 1.0,
        'num_mc_samples': 15,
        'ensemble_weight_decay': 0.01,
        'num_nodes': num_nodes,
    }
    
def load_pretrained_model(pretrained_path, device):
    """Load pre-trained model from transfer learning"""
    if pretrained_path is None or not os.path.exists(pretrained_path):
        return None, None
    
    print(f"Loading pre-trained model from: {pretrained_path}")
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Extract model state and config
        model_state = checkpoint['model_state_dict']
        pretrained_config = checkpoint['config']
        transferred_layers = checkpoint.get('transferred_layers', [])
        pretrained_domains = checkpoint.get('pretrained_domains', [])
        
        print(f"Loaded pre-trained model with:")
        print(f"  - Transferred layers: {len(transferred_layers)}")
        print(f"  - Pre-trained domains: {pretrained_domains}")
        print(f"  - Transfer strategy: {checkpoint.get('transfer_strategy', 'N/A')}")
        
        return model_state, pretrained_config
        
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None, None

def initialize_model_with_pretrained(hp, pretrained_state, pretrained_config, device):
    """Initialize model with pre-trained weights where compatible"""
    model = SpatioTemporalEnsemble(hp).to(device)
    
    if pretrained_state is not None:
        print("Initializing model with pre-trained weights...")
        model_state = model.state_dict()
        
        # Count transferred layers
        transferred_count = 0
        skipped_count = 0
        
        for name, param in pretrained_state.items():
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred_count += 1
                else:
                    skipped_count += 1
                    print(f"  Shape mismatch - {name}: {param.shape} vs {model_state[name].shape}")
            else:
                skipped_count += 1
        
        model.load_state_dict(model_state)
        print(f"Transferred {transferred_count} layers, skipped {skipped_count} layers")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Ensemble Hyperparameter Optimization')
    parser.add_argument('--pretrained_model', type=str, default='/transfer_learning_pretrain/transferred_model_for_hp_search.pt', help='path to pre-trained model from transfer learning')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')  # Increased to 200
    parser.add_argument('--iterations', type=int, default=20, help='number of random search iterations')  # Increased to 60
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    create_directories()
    set_random_seed(123)
    
    
    # Load pre-trained model if provided
    pretrained_state, pretrained_config = load_pretrained_model(args.pretrained_model, device)
    
    if pretrained_state is not None:
        print("Using pre-trained model as initialization for hyperparameter optimization")
    else:
        print("No pre-trained model provided, using random initialization")
    
    print("Loading data with compatible parameters...")
    try:
        data = DataLoaderEnsemble(args.data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=12)
        print(f"Data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Starting hyperparameter optimization...")
    print(f"Training sequences: {data.train[0].shape[0]}")
    print(f"Validation sequences: {data.valid[0].shape[0]}")
    print(f"Test sequences: {data.test[0].shape[0]}")
    print(f"Number of nodes: {data.m}")
    print(f"Data output length: {data.out_len}")
    
    best_val_rrse = float('inf')
    best_val_rae = float('inf')
    best_hp = None
    best_model_state = None
    
    successful_iterations = 0
    all_results = []
    
    # Adaptive sampling strategy
    best_configs_history = []
    
    for iteration in range(args.iterations):
        print(f"\n Iteration {iteration + 1}/{args.iterations} ")
        
        # Adaptive sampling based on iteration phase
        if iteration < 20:
            # Phase 1: Broad exploration
            hp = sample_hyperparameters(search_space, validation_rules, data)
        elif iteration < 40:
            # Phase 2: Focused refinement - sample near best-performing configs
            if best_configs_history:
                # Use best config as base and perturb
                base_hp = random.choice(best_configs_history[-5:])  # Use recent best configs
                hp = perturb_hyperparameters(base_hp, search_space, validation_rules, data)
            else:
                hp = sample_hyperparameters(search_space, validation_rules, data)
        else:
            # Phase 3: Fine-tuning - narrow ranges around optimal architecture
            if best_hp is not None:
                hp = fine_tune_hyperparameters(best_hp, search_space, validation_rules, data)
            else:
                hp = sample_hyperparameters(search_space, validation_rules, data)
        
        print(f"Hyperparameters: {hp}")
        
        try:
            # Create model with pre-trained weights
            print("Initializing ensemble model with pre-trained weights...")
            model = initialize_model_with_pretrained(hp, pretrained_state, pretrained_config, device)
            
            # Rest of the training code remains the same...
            criterion = nn.MSELoss()
            evaluateL2 = nn.MSELoss()
            evaluateL1 = nn.L1Loss()
            
            # Configure optimizer based on hyperparameters
            if hp['optimizer_type'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], 
                                    weight_decay=hp['weight_decay'])
            else:
                optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'],
                                    weight_decay=hp['weight_decay'])
                
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=hp['scheduler_patience'], 
                                                           factor=hp['scheduler_factor']) 
            # Training loop
            best_iter_val_rrse = float('inf')
            best_iter_val_rae = float('inf')
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
                    best_model_state_iter = model.state_dict().copy()
                    patience_counter = 0
                    print(f"  -> New best for iteration: RRSE={val_rrse:.6f}, RAE={val_rae:.6f}")
                else:
                    patience_counter += 1
                
                # Early stopping for this iteration
                if patience_counter >= 15:  # Slightly more patience for 200 epochs
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Store results for this iteration
            iteration_result = {
                'iteration': iteration + 1,
                'hyperparameters': hp,
                'best_train_loss': train_loss,
                'best_val_rrse': best_iter_val_rrse,
                'best_val_rae': best_iter_val_rae,
                'epochs_trained': epoch
            }
            all_results.append(iteration_result)
            
            # Update global best
            if (not math.isinf(best_iter_val_rrse) and not math.isnan(best_iter_val_rrse) and 
                best_iter_val_rrse < best_val_rrse and best_model_state_iter is not None):
                best_val_rrse = best_iter_val_rrse
                best_val_rae = best_iter_val_rae
                best_hp = hp.copy()
                best_model_state = best_model_state_iter
                successful_iterations += 1
                
                # Store for adaptive sampling
                best_configs_history.append(hp.copy())
                if len(best_configs_history) > 10:  # Keep only recent best
                    best_configs_history.pop(0)
                    
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
        # Save best hyperparameters
        with open('model/Ensemble/hp.txt', 'w') as f:
            json.dump(best_hp, f, indent=2, cls=NumpyEncoder)
        
        # Save detailed optimization results
        with open('model/Ensemble/hyperparameter_optimization_results.json', 'w') as f:
            json.dump({
                'all_results': all_results,
                'best_hyperparameters': best_hp,
                'best_validation_rrse': best_val_rrse,
                'best_validation_rae': best_val_rae,
                'successful_iterations': successful_iterations,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, cls=NumpyEncoder)
        
        # Load best model for final evaluation
        model = SpatioTemporalEnsemble(best_hp).to(device)
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
        
        model_save_data = {
            'model_state_dict': best_model_state,
            'hyperparameters': best_hp,
            'validation_rrse': val_rrse,
            'validation_rae': val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'successful_iterations': successful_iterations,
            'timestamp': datetime.now().isoformat(),
            'pretrained_initialization': args.pretrained_model is not None,
            'pretrained_model_path': args.pretrained_model
            }
        
        torch.save(model_save_data, 'model/Ensemble/best_hyperparameter_model.pt')
        print(f"\n OPTIMIZATION COMPLETE ")
        print(f"Best model saved as 'model/Ensemble/best_hyperparameter_model.pt'")
        print(f"Final Validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if data.test[0].size(0) > 0:
            print(f"Final Testing - RRSE: {test_rrse:.6f}, RAE: {test_rae:.6f}")
        
    else:
        # Fallback: create and save a model with default parameters
        print("\n CREATING FALLBACK MODEL ")
        fallback_hp = get_conservative_config(data.m)
        
        with open('model/Ensemble/hp.txt', 'w') as f:
            json.dump(fallback_hp, f, indent=2, cls=NumpyEncoder)
        
        model = SpatioTemporalEnsemble(fallback_hp).to(device)
        
        # Create some basic plots even with fallback model
        if data.valid[0].size(0) > 0:
            evaluate_comprehensive(data, data.valid[0], data.valid[1], model, 
                                 nn.MSELoss(), nn.L1Loss(), fallback_hp['batch_size'], True, device, "Validation")
            print("Fallback model evaluated on validation data and plots created.")
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': fallback_hp,
            'is_fallback': True,
            'timestamp': datetime.now().isoformat()
        }, 'model/Ensemble/best_hyperparameter_model.pt')
        
        print("Fallback model saved as 'model/Ensemble/best_hyperparameter_model.pt'")

    print(f"\nHyperparameter optimization completed.")
    print(f"Plots with confidence intervals saved in: model/Ensemble/Validation/ and model/Ensemble/Testing/")
    print(f"Model saved as: model/Ensemble/best_hyperparameter_model.pt")
    print(f"Hyperparameters saved as: model/Ensemble/hp.txt")
    print(f"Detailed results saved as: model/Ensemble/hyperparameter_optimization_results.json")

# Helper functions for adaptive sampling
def perturb_hyperparameters(base_hp, search_space, validation_rules, data, max_attempts=50):
    """Perturb a base configuration for focused refinement"""
    for attempt in range(max_attempts):
        hp = base_hp.copy()
        
        # Perturb key parameters with moderate changes
        if random.random() < 0.7:  # 70% chance to perturb each parameter
            hp['vit_embed_dim'] = random.choice([max(64, hp['vit_embed_dim'] - 32), 
                                               hp['vit_embed_dim'], 
                                               min(160, hp['vit_embed_dim'] + 32)])
        
        if random.random() < 0.7:
            hp['vit_hidden_dim'] = random.choice([max(128, hp['vit_hidden_dim'] - 64),
                                                hp['vit_hidden_dim'],
                                                min(320, hp['vit_hidden_dim'] + 64)])
        
        if random.random() < 0.6:
            hp['learning_rate'] = random.choice([
                max(0.0001, hp['learning_rate'] / 2),
                hp['learning_rate'],
                min(0.002, hp['learning_rate'] * 1.5)
            ])
        
        # Ensure we're still within search space bounds
        for key, values in search_space.items():
            if key in hp and hp[key] not in values:
                # Find closest valid value
                closest = min(values, key=lambda x: abs(x - hp[key]))
                hp[key] = closest
        
        # Apply validation rules
        if all(rule(hp) for rule in validation_rules.values()):
            return hp
    
    # Fallback to original base configuration
    return base_hp

def fine_tune_hyperparameters(base_hp, search_space, validation_rules, data, max_attempts=50):
    """Fine-tune around the best configuration with small changes"""
    for attempt in range(max_attempts):
        hp = base_hp.copy()
        
        # Make very small perturbations for fine-tuning
        if random.random() < 0.8:
            # Small changes to learning rate
            lr_options = [base_hp['learning_rate']]
            if base_hp['learning_rate'] > min(search_space['learning_rate']):
                lr_options.append(max(min(search_space['learning_rate']), base_hp['learning_rate'] * 0.8))
            if base_hp['learning_rate'] < max(search_space['learning_rate']):
                lr_options.append(min(max(search_space['learning_rate']), base_hp['learning_rate'] * 1.2))
            hp['learning_rate'] = random.choice(lr_options)
        
        if random.random() < 0.5:
            # Small changes to dropout
            hp['dropout'] = random.choice([
                max(0.1, base_hp['dropout'] - 0.05),
                base_hp['dropout'],
                min(0.3, base_hp['dropout'] + 0.05)
            ])
        
        # Apply validation rules
        if all(rule(hp) for rule in validation_rules.values()):
            return hp
    
    # Fallback to original base configuration
    return base_hp

if __name__ == "__main__":
    main()