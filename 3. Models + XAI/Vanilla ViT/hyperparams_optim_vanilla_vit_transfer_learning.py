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

class SimplePatchEmbedding(nn.Module):
    """Simple patch embedding for time series (fixed patch-size selection + checks)"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = int(seq_len)
        self.requested_patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)

        # compute compatible patch size (prefer requested if compatible)
        self.patch_size = self._compatible_patch_size(self.seq_len, self.requested_patch_size)
        self.num_patches = self.seq_len // self.patch_size

        print(f"[PatchEmbedding] seq_len={self.seq_len}, requested_patch_size={self.requested_patch_size}, "
              f"final_patch_size={self.patch_size}, num_patches={self.num_patches}, in_channels={self.in_channels}")

        # linear projection input dim = patch_size * in_channels
        self.projection = nn.Linear(self.patch_size * self.in_channels, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim) * 0.02
        )

    def _compatible_patch_size(self, seq_len, desired):
        # get divisors
        divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
        if desired in divisors:
            return desired
        # choose largest divisor <= desired (keeps patches small / sensible)
        le = [d for d in divisors if d <= desired]
        if le:
            return max(le)
        # otherwise choose smallest divisor > desired
        gt = [d for d in divisors if d > desired]
        if gt:
            return min(gt)
        # fallback
        return 1

    def forward(self, x):
        # expected x shape: (batch_size, seq_len, num_nodes/in_channels)
        batch_size = x.shape[0]
        assert x.shape[1] == self.seq_len, f"input seq_len ({x.shape[1]}) != expected ({self.seq_len})"
        # unfold -> (batch_size, num_patches, in_channels, patch_size) or variant depending on layout
        # Here x is (B, seq_len, num_nodes), we want windows along seq dim of length patch_size
        # Using unfold on dim=1
        x_patches = x.unfold(1, self.patch_size, self.patch_size)  # (B, num_patches, in_channels, patch_size)
        # If unfold returns (..., in_channels, patch_size), we want to flatten last two dims
        x_patches = x_patches.contiguous().view(batch_size, self.num_patches, self.in_channels * self.patch_size)

        # sanity check dims before linear
        in_feat = self.patch_size * self.in_channels
        if self.projection.in_features != in_feat:
            raise RuntimeError(
                f"Projection in_features ({self.projection.in_features}) != computed patch dim ({in_feat}). "
                "This indicates mismatch between patch_size/in_channels and projection layer."
            )

        # project
        x_emb = self.projection(x_patches)  # (B, num_patches, embed_dim)

        # cls token + positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x_out = torch.cat((cls_tokens, x_emb), dim=1)  # (B, num_patches+1, D)
        x_out = x_out + self.position_embeddings

        return x_out


class SimpleVisionTransformer(nn.Module):
    """Simplified Vision Transformer for time series forecasting (with fixed patch-size selection)"""
    def __init__(self, config):
        super().__init__()
        # ensure ints
        cfg = dict(config)
        cfg['sequence_length'] = int(cfg['sequence_length'])
        cfg['patch_size'] = int(cfg['patch_size'])
        cfg['embed_dim'] = int(cfg['embed_dim'])
        cfg['num_heads'] = int(cfg['num_heads'])
        cfg['hidden_dim'] = int(cfg['hidden_dim'])
        cfg['num_layers'] = int(cfg['num_layers'])
        cfg['batch_size'] = int(cfg['batch_size'])
        cfg['forecast_horizon'] = int(cfg['forecast_horizon'])
        cfg['num_nodes'] = int(cfg['num_nodes'])
        cfg['dropout'] = float(cfg.get('dropout', 0.0))

        # compute compatible patch size (use same helper semantics as embedding)
        actual_patch_size = self._get_compatible_patch_size(cfg['sequence_length'], cfg['patch_size'])
        actual_num_patches = cfg['sequence_length'] // actual_patch_size

        print(f" sequence_length={cfg['sequence_length']}, requested_patch_size={cfg['patch_size']}, "
              f"actual_patch_size={actual_patch_size}, num_patches={actual_num_patches}, num_nodes={cfg['num_nodes']}")

        self.config = cfg

        self.patch_embed = SimplePatchEmbedding(
            seq_len=cfg['sequence_length'],
            patch_size=actual_patch_size,
            in_channels=cfg['num_nodes'],
            embed_dim=cfg['embed_dim']
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['embed_dim'],
            nhead=self._get_compatible_heads(cfg['embed_dim'], cfg['num_heads']),
            dim_feedforward=cfg['hidden_dim'],
            dropout=cfg['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])

        self.forecast_head = nn.Sequential(
            nn.Linear(cfg['embed_dim'], cfg['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden_dim'], cfg['forecast_horizon'] * cfg['num_nodes'])
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _get_compatible_patch_size(self, seq_len, desired):
        divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
        if desired in divisors:
            return desired
        le = [d for d in divisors if d <= desired]
        if le:
            return max(le)
        gt = [d for d in divisors if d > desired]
        if gt:
            return min(gt)
        return 1

    def _get_compatible_heads(self, embed_dim, desired_heads):
        divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
        if desired_heads in divisors:
            return desired_heads
        # pick the largest divisor <= desired_heads, else smallest > desired_heads, else 1
        le = [d for d in divisors if d <= desired_heads]
        if le:
            return max(le)
        gt = [d for d in divisors if d > desired_heads]
        if gt:
            return min(gt)
        return 1

    def forward(self, x):
        # x: (B, seq_len, num_nodes)
        x = self.patch_embed(x)             # (B, num_patches+1, D)
        x = self.transformer(x)             # (B, num_patches+1, D)
        cls_token = x[:, 0]                 # (B, D)
        forecast = self.forecast_head(cls_token)
        forecast = forecast.view(-1, self.config['forecast_horizon'], self.config['num_nodes'])
        return forecast


class SimpleDataLoader:
    """Simplified data loader for time series data"""
    def __init__(self, data_file, train_ratio, valid_ratio, device, horizon, seq_len, normalize, out_len):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.device = device
        self.horizon = horizon
        self.seq_len = seq_len
        self.normalize = normalize
        self.out_len = out_len
        
        self.read_data()
        self.split_data()
        
    def read_data(self): 
        """Read and normalize data"""
        print(f"Loading data from: {self.data_file}")

        self.data = pd.read_csv(self.data_file)
        self.raw_data = self.data.values
        print(f"Raw data shape: {self.raw_data.shape}")

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
        
    def split_data(self):
        """Build sliding-window sequences from the entire normalized dataset, then split sequences."""
        total_samples = len(self.normalized_data)
        required = self.seq_len + self.out_len
        if total_samples < required:
            print(f"Warning: Not enough total data to form any sequences. Need {required}, have {total_samples}")
            self.train = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            self.valid = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            self.test  = (torch.zeros(0, self.seq_len, self.m), torch.zeros(0, self.out_len, self.m))
            return

        # Build all sequences by sliding window across the entire normalized dataset
        n_seq = total_samples - required + 1
        X_all = np.zeros((n_seq, self.seq_len, self.m))
        Y_all = np.zeros((n_seq, self.out_len, self.m))
        for i in range(n_seq):
            X_all[i] = self.normalized_data[i:i+self.seq_len]
            Y_all[i] = self.normalized_data[i+self.seq_len:i+self.seq_len+self.out_len]

        # Now split sequences into train/valid/test by ratio (applied on sequence count)
        n_train = int(n_seq * self.train_ratio)
        n_valid = int(n_seq * self.valid_ratio)
        # ensure at least one sample goes to test if possible
        n_test = n_seq - n_train - n_valid
        if n_test < 0:
            # adjust to ensure non-negative
            n_valid = max(0, n_seq - n_train)
            n_test = 0

        def to_tensors(arr):
            return torch.from_numpy(arr).float()

        self.train = (to_tensors(X_all[:n_train]), to_tensors(Y_all[:n_train]))
        self.valid = (to_tensors(X_all[n_train:n_train + n_valid]), to_tensors(Y_all[n_train:n_train + n_valid]))
        self.test  = (to_tensors(X_all[n_train + n_valid:]), to_tensors(Y_all[n_train + n_valid:]))

        print(f"Total sequences: {n_seq}")
        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")
        
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
            return
            
        if shuffle:
            indices = torch.randperm(X.size(0))
            X = X[indices]
            Y = Y[indices]
            
        for i in range(0, X.size(0), batch_size):
            yield X[i:i+batch_size], Y[i:i+batch_size]

# TRANSFER LEARNING CLASSES AND FUNCTIONS

class TransferLearningManager:
    """Manages transfer learning from pre-trained models"""
    
    def __init__(self, pretrained_model_path, device):
        self.pretrained_model_path = pretrained_model_path
        self.device = device
    
    def load_pretrained_model(self):
        """Load the pre-trained model"""
        if not os.path.exists(self.pretrained_model_path):
            print(f"Warning: Pre-trained model not found at {self.pretrained_model_path}")
            return None
        
        try:
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            return checkpoint
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return None
    
    def transfer_weights(self, target_model, transfer_strategy='full'):
        """Transfer weights from pre-trained model to target model; summarize mismatches."""
        pretrained_checkpoint = self.load_pretrained_model()
        
        if pretrained_checkpoint is None:
            print("No pre-trained model available, using random initialization")
            return target_model, []
        
        pretrained_state = pretrained_checkpoint.get('model_state_dict', pretrained_checkpoint)
        target_state = target_model.state_dict()
        
        transferred_layers = []
        mismatch_count = 0
        total_checked = 0
        
        if transfer_strategy == 'full':
            items = pretrained_state.items()
        elif transfer_strategy == 'encoder_only':
            items = ((k,v) for k,v in pretrained_state.items() if any(key in k for key in ['patch_embed', 'transformer']))
        elif transfer_strategy == 'embedding_only':
            items = ((k,v) for k,v in pretrained_state.items() if 'patch_embed' in k)
        else:
            items = []

        for name, param in items:
            if name in target_state:
                total_checked += 1
                if target_state[name].shape == param.shape:
                    target_state[name] = param
                    transferred_layers.append(name)
                else:
                    mismatch_count += 1
                    # skip silently; avoid noisy per-layer prints
            # else: not present in target, skip
		
        target_model.load_state_dict(target_state)
        print(f"Transferred {len(transferred_layers)} layers (checked {total_checked} candidates, {mismatch_count} shape mismatches) using strategy: {transfer_strategy}")
        
        return target_model, transferred_layers

def create_directories():
    """Create all necessary directories for plots and models"""
    directories = [
        'model/ViT_Transfer_Optim/Validation',
        'model/ViT_Transfer_Optim/Testing', 
        'model/ViT_Transfer_Optim/plots',
        'model/ViT_Transfer_Optim/data'
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
    with open(f'model/ViT_Transfer_Optim/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')
        f.close()

def plot_predicted_actual(predicted, actual, title, type):
    """Plot predicted vs actual curves"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    M = []
    
    # Create proper time range from 2011 to 2019
    for year in range(11, 20):  # 2011 to 2019
        for month in months:
            if year == 11 and month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']:
                continue
            M.append(month + '-' + str(year))
    
    # Extend if needed
    if len(predicted) > len(M):
        for year in range(20, 25):
            for month in months:
                M.append(month + '-' + str(year))
    
    M2 = []
    p = []
    
    if type == 'Testing':
        M = M[-len(predicted):] if len(predicted) <= len(M) else M
    else:
        if len(predicted) <= len(M):
            M = M[:len(predicted)]
        else:
            current_len = len(M)
            for i in range(len(predicted) - current_len):
                M.append(f"Month_{i+1}")
    
    # Select every 3rd month for x-axis labels
    for index, value in enumerate(M):
        if index % 3 == 0:
            M2.append(value)
            p.append(index + 1)
    
    if len(M2) < 4 and len(M) > 0:
        M2 = [M[i] for i in range(0, len(M), max(1, len(M)//4))]
        p = [i+1 for i in range(0, len(M), max(1, len(M)//4))]

    x = range(1, len(predicted) + 1)
    plt.figure(figsize=(14, 6))
    plt.plot(x, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted, '--', color='purple', label='Predicted', linewidth=2)
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(f"{title} - {type}", y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Time", fontsize=15)
    
    if p and M2:
        plt.xticks(ticks=p, labels=M2, rotation=45, fontsize=11)
    else:
        plt.xticks(fontsize=11)
        
    plt.yticks(fontsize=11)
    
    title = title.replace('/', '_')
    plt.savefig(f'model/ViT_Transfer_Optim/{type}/{title}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/ViT_Transfer_Optim/{type}/{title}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()
    print(f"  Saved plot: model/ViT_Transfer_Optim/{type}/{title}_{type}.png")

def calculate_rrse_rae_comprehensive(predict, test, scale):
    """Comprehensive RRSE and RAE calculation (robust to zero denominators)"""
    if predict.numel() == 0 or test.numel() == 0:
        print("Warning: Empty tensors in metric calculation")
        return float('inf'), float('inf')
    
    try:
        eps = 1e-8  # small value to avoid division by zero

        # Denormalize the data
        predict_denorm = predict * scale
        test_denorm = test * scale
        
        # Flatten all dimensions for overall metrics
        predict_flat = predict_denorm.flatten()
        test_flat = test_denorm.flatten()
        
        # Calculate numerator for RRSE (root sum squared errors)
        squared_errors = (test_flat - predict_flat) ** 2
        sum_squared_errors = torch.sum(squared_errors)
        root_sum_squared_errors = torch.sqrt(sum_squared_errors + 0.0)
        
        # Calculate denominator for RRSE (root sum squared deviations)
        test_mean = torch.mean(test_flat)
        squared_deviations = (test_flat - test_mean) ** 2
        sum_squared_deviations = torch.sum(squared_deviations)
        root_sum_squared_deviations = torch.sqrt(sum_squared_deviations + eps)
        
        # Calculate RRSE robustly
        rrse = root_sum_squared_errors / (root_sum_squared_deviations + eps)
        
        # Calculate numerator for RAE
        absolute_errors = torch.abs(test_flat - predict_flat)
        sum_absolute_errors = torch.sum(absolute_errors)
        
        # Calculate denominator for RAE (absolute deviations from mean)
        absolute_deviations = torch.abs(test_flat - test_mean)
        sum_absolute_deviations = torch.sum(absolute_deviations)
        
        # Calculate RAE robustly
        rae = sum_absolute_errors / (sum_absolute_deviations + eps)
        
        # Clamp and convert to python floats
        if isinstance(rrse, torch.Tensor):
            rrse = rrse.item()
        if isinstance(rae, torch.Tensor):
            rae = rae.item()
        
        # Guard against NaN/Infs
        if not math.isfinite(rrse):
            rrse = float('inf')
        if not math.isfinite(rae):
            rae = float('inf')
        
        return rrse, rae
        
    except Exception as e:
        print(f"Error in comprehensive metric calculation: {e}")
        return float('inf'), float('inf')

def evaluate_simple(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot, device, eval_type="Validation"):
    """Simple evaluation function without Bayesian components"""
    model.eval()  # No dropout during evaluation
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    all_predictions = []
    all_targets = []

    if X.size(0) == 0:
        print(f"CRITICAL: No {eval_type} data available for evaluation")
        return float('inf'), float('inf')

    print(f"Evaluating on {X.size(0)} {eval_type.lower()} sequences...")

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(data.get_batches(X, Y, batch_size, False)):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            output = model(X_batch)
            
            # Ensure prediction horizon matches target horizon
            min_horizon = min(output.shape[1], Y_batch.shape[1])
            output = output[:, :min_horizon, :]
            
            # Store predictions and targets
            scale_batch = data.scale.expand(output.size(0), output.size(1), data.m).to(device)
            predictions_denorm = output * scale_batch
            targets_denorm = Y_batch[:, :min_horizon, :] * scale_batch

            all_predictions.append(predictions_denorm.cpu())
            all_targets.append(targets_denorm.cpu())

            # Calculate losses
            total_loss += evaluateL2(output, Y_batch[:, :min_horizon, :]).item()
            total_loss_l1 += evaluateL1(output, Y_batch[:, :min_horizon, :]).item()
            n_samples += (output.size(0) * output.size(1) * data.m)

    if not all_predictions:
        print(f"CRITICAL: No predictions were generated during {eval_type.lower()} evaluation")
        return float('inf'), float('inf')

    try:
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        print(f"{eval_type} evaluation completed: {all_predictions_tensor.shape} predictions")
        
        # Calculate comprehensive metrics
        rrse, rae = calculate_rrse_rae_comprehensive(all_predictions_tensor, all_targets_tensor, data.scale)
        
        print(f"Calculated {eval_type.lower()} metrics - RRSE: {rrse:.6f}, RAE: {rae:.6f}")
        
        # Plot results if requested
        if is_plot and all_predictions_tensor.numel() > 0:
            plot_evaluation_results_simple(all_predictions_tensor, all_targets_tensor, data, eval_type)
            
        return rrse, rae
        
    except Exception as e:
        print(f"Error in {eval_type.lower()} evaluation aggregation: {e}")
        return float('inf'), float('inf')

def plot_evaluation_results_simple(predictions, targets, data, eval_type): 
    """Plot evaluation results for sample nodes without confidence intervals"""
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Plot first ALL nodes (columns)
    num_nodes_to_plot = min(645, data.m)
    print(f"Creating {num_nodes_to_plot} {eval_type.lower()} plots...")

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

            # Save metrics and plots
            save_metrics_1d(torch.from_numpy(pred_curve), torch.from_numpy(target_curve), node_name, eval_type)
            plot_predicted_actual(pred_curve, target_curve, node_name, eval_type)

def train_simple(data, X, Y, model, criterion, optimizer, batch_size, device):
    """Simple training function (avg loss per batch, guarded)"""
    model.train()
    total_loss = 0.0
    batch_count = 0

    if X.size(0) == 0:
        print("CRITICAL: No training data available")
        return float('inf')

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
        output_denorm = output * scale
        target_denorm = Y_batch * scale

        loss = criterion(output_denorm, target_denorm)
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss encountered (batch {batch_idx}). Skipping batch.")
            continue

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")

    if batch_count == 0:
        print("CRITICAL: No valid batches processed during training")
        return float('inf')

    avg_loss = total_loss / batch_count
    print(f"  Training completed: {batch_count} batches, Avg Loss (per-batch): {avg_loss:.6f}")
    return avg_loss

def sample_simple_hyperparameters(search_space, num_nodes):
    """Sample hyperparameters for simple ViT"""
    max_attempts = 100
    for attempt in range(max_attempts):
        hp = {}
        for key, values in search_space.items():
            hp[key] = random.choice(values)
        
        # Apply compatibility constraints
        if hp['embed_dim'] % hp['num_heads'] != 0:
            compatible_heads = [h for h in search_space['num_heads'] if hp['embed_dim'] % h == 0]
            if compatible_heads:
                hp['num_heads'] = random.choice(compatible_heads)
            else:
                continue
        
        if hp['sequence_length'] % hp['patch_size'] != 0:
            compatible_patches = [p for p in search_space['patch_size'] if hp['sequence_length'] % p == 0]
            if compatible_patches:
                hp['patch_size'] = random.choice(compatible_patches)
            else:
                continue
        
        if hp['hidden_dim'] < hp['embed_dim']:
            compatible_hidden = [h for h in search_space['hidden_dim'] if h >= hp['embed_dim']]
            if compatible_hidden:
                hp['hidden_dim'] = random.choice(compatible_hidden)
            else:
                continue
        
        # All constraints satisfied
        return hp
    
    # Fallback
    print("Warning: Could not find fully compatible hyperparameters, using fallback")
    fallback_hp = {
        'sequence_length': 12,
        'patch_size': 4,
        'embed_dim': 128,
        'num_heads': 8,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 8,
        'forecast_horizon': 12,
    }
    return fallback_hp

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Hyperparameter Optimization')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--iterations', type=int, default=60, help='number of random search iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    # Transfer learning specific arguments
    parser.add_argument('--pretrained_model', type=str, default='transfer_learning_pretrain/transferred_model_finetuned.pt', 
                       help='path to pre-trained model for transfer learning')
    parser.add_argument('--transfer_strategy', type=str, default='full', 
                       choices=['full', 'encoder_only', 'embedding_only', 'none'],
                       help='strategy for weight transfer from pre-trained model')
    parser.add_argument('--finetune_lr', type=float, default=0.0001, 
                       help='learning rate for fine-tuning (typically lower than normal training)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create directories first
    create_directories()
    
    # Set fixed random seed
    set_random_seed(123)
    
    print("Loading data...")
    try:
        data = SimpleDataLoader(args.data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=12)
        print(f"Data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize transfer learning manager
    transfer_manager = TransferLearningManager(args.pretrained_model, device)
    
    search_space = {
        'sequence_length': [24],
        'patch_size': [2, 4, 6, 8],
        'embed_dim': [64, 128, 256],
        'num_heads': [4, 8],
        'hidden_dim': [128, 256, 512],
        'num_layers': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [args.finetune_lr],  # Use fine-tuning learning rate
        'weight_decay': [0.0, 0.001, 0.01],
        'batch_size': [4, 8, 16],
        'forecast_horizon': [12],
    }
    
    print("Starting hyperparameter optimization with transfer learning...")
    print(f"Pre-trained model: {args.pretrained_model}")
    print(f"Transfer strategy: {args.transfer_strategy}")
    print(f"Fine-tuning learning rate: {args.finetune_lr}")
    print(f"Training sequences: {data.train[0].shape[0]}")
    print(f"Validation sequences: {data.valid[0].shape[0]}")
    print(f"Test sequences: {data.test[0].shape[0]}")
    print(f"Number of nodes: {data.m}")

    best_val_rrse = float('inf')
    best_val_rae = float('inf')
    best_hp = None
    best_model_state = None
    best_loss = float('inf')
    best_transferred_layers = []
    
    successful_iterations = 0
    all_results = []
    
    for iteration in range(args.iterations):
        print(f"\n Iteration {iteration + 1}/{args.iterations} ")
        
        hp = sample_simple_hyperparameters(search_space, data.m)
        hp['num_nodes'] = data.m
        
        print(f"Hyperparameters: {hp}")
        
        try:
            if hp.get('sequence_length', data.seq_len) != getattr(data, 'seq_len', None) or \
               hp.get('forecast_horizon', data.out_len) != getattr(data, 'out_len', None):
                print("Reinitializing data with new sequence/forecast lengths to match model...")
                # Recreate data with same train/valid ratios and normalization behavior
                data = SimpleDataLoader(args.data, data.train_ratio, data.valid_ratio, device,
                                        horizon=1, seq_len=hp['sequence_length'], normalize=2, out_len=hp['forecast_horizon'])
                print(f"Reinitialized data: train seqs={data.train[0].shape[0]}, valid seqs={data.valid[0].shape[0]}, test seqs={data.test[0].shape[0]}")
                
                # If new split yields no sequences, skip this sampled hp
                if data.train[0].size(0) == 0 or data.valid[0].size(0) == 0:
                    print("Skipping iteration due to insufficient sequences after reinitialization.")
                    continue
        except Exception as e:
            print(f"Failed to reinitialize data for hp change: {e}")
            continue
        
        try:
            # Create model
            print("Initializing model...")
            model = SimpleVisionTransformer(hp).to(device)
            
            # Apply transfer learning if requested
            if args.transfer_strategy != 'none':
                model, transferred_layers = transfer_manager.transfer_weights(
                    model, args.transfer_strategy
                )
            else:
                transferred_layers = []
                print("Skipping transfer learning (none strategy)")
            
            # Simple training configuration
            criterion = nn.MSELoss()
            evaluateL2 = nn.MSELoss()
            evaluateL1 = nn.L1Loss()
            
            optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
            
            # Training loop
            best_iter_val_rrse = float('inf')
            best_iter_val_rae = float('inf')
            best_iter_loss = float('inf')
            patience_counter = 0
            best_model_state_iter = None
            
            print("Starting training...")
            for epoch in range(1, args.epochs + 1):
                # Training phase
                train_loss = train_simple(data, data.train[0], data.train[1], model, criterion, optimizer, hp['batch_size'], device)
                
                # Validation phase
                val_rrse, val_rae = evaluate_simple(data, data.valid[0], data.valid[1], model, evaluateL2, evaluateL1, hp['batch_size'], False, device, "Validation")
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val RRSE: {val_rrse:.6f}, Val RAE: {val_rae:.6f}')
                
                # Save best model for this iteration
                if (not math.isinf(val_rrse) and not math.isnan(val_rrse) and 
                    val_rrse < best_iter_val_rrse):
                    best_iter_val_rrse = val_rrse
                    best_iter_val_rae = val_rae
                    best_iter_loss = train_loss
                    best_model_state_iter = model.state_dict().copy()
                    patience_counter = 0
                    print(f"  -> New best for iteration: RRSE={val_rrse:.6f}, RAE={val_rae:.6f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= 10:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Store results for this iteration
            iteration_result = {
                'iteration': iteration + 1,
                'hyperparameters': hp,
                'transferred_layers': transferred_layers,
                'best_train_loss': best_iter_loss,
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
                best_loss = best_iter_loss
                best_hp = hp.copy()
                best_model_state = best_model_state_iter
                best_transferred_layers = transferred_layers
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
        # Save best hyperparameters
        with open('model/ViT_Transfer_Optim/hp.txt', 'w') as f:
            json.dump(best_hp, f, indent=2, cls=NumpyEncoder)
        
        # Load best model for final evaluation
        model = SimpleVisionTransformer(best_hp).to(device)
        model.load_state_dict(best_model_state)
        
        # Final validation evaluation with plots
        print("Final validation evaluation...")
        val_rrse, val_rae = evaluate_simple(data, data.valid[0], data.valid[1], model, 
                                          nn.MSELoss(), nn.L1Loss(), best_hp['batch_size'], True, device, "Validation")
        
        # Final test evaluation with plots
        if data.test[0].size(0) > 0:
            print("Final test evaluation...")
            test_rrse, test_rae = evaluate_simple(data, data.test[0], data.test[1], model, 
                                                nn.MSELoss(), nn.L1Loss(), best_hp['batch_size'], True, device, "Testing")
        else:
            print("No test data available for evaluation")
            test_rrse, test_rae = float('inf'), float('inf')
        
        # Save the best model
        model_save_data = {
            'model_state_dict': best_model_state,
            'hyperparameters': best_hp,
            'transferred_layers': best_transferred_layers,
            'transfer_strategy': args.transfer_strategy,
            'pretrained_model': args.pretrained_model,
            'validation_rrse': val_rrse,
            'validation_rae': val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'successful_iterations': successful_iterations,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_save_data, 'model/ViT_Transfer_Optim/best_hyperparameter_model.pt')
        
        # Save training configuration and results
        training_summary = {
            'hyperparameters': best_hp,
            'transfer_strategy': args.transfer_strategy,
            'pretrained_model': args.pretrained_model,
            'transferred_layers_count': len(best_transferred_layers),
            'final_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.raw_data.shape,
            'device': str(device),
            'best_val_rrse': best_val_rrse,
            'best_val_rae': best_val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'successful_iterations': successful_iterations,
            'total_iterations': args.iterations
        }
        
        with open('model/ViT_Transfer_Optim/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
        print("Training summary saved as 'model/ViT_Transfer_Optim/training_summary.json'")
        
        print(f"\n OPTIMIZATION COMPLETE ")
        print(f"Best model saved as 'model/ViT_Transfer_Optim/best_hyperparameter_model.pt'")
        print(f"Final Validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if data.test[0].size(0) > 0:
            print(f"Final Testing - RRSE: {test_rrse:.6f}, RAE: {test_rae:.6f}")
        print(f"Transferred layers: {len(best_transferred_layers)}")
        
    else:
        # Fallback: create and save a model with default parameters
        print("\n CREATING FALLBACK MODEL ")
        default_hp = {
            'sequence_length': 24,
            'patch_size': 4,
            'embed_dim': 128,
            'num_heads': 8,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': args.finetune_lr,
            'weight_decay': 0.01,
            'batch_size': 8,
            'forecast_horizon': 12,
            'num_nodes': data.m
        }
        
        with open('model/ViT_Transfer_Optim/hp.txt', 'w') as f:
            json.dump(default_hp, f, indent=2, cls=NumpyEncoder)
        
        model = SimpleVisionTransformer(default_hp).to(device)
        
        # Apply transfer learning to fallback model if requested
        if args.transfer_strategy != 'none':
            model, transferred_layers = transfer_manager.transfer_weights(model, args.transfer_strategy)
            print(f"Transferred {len(transferred_layers)} layers to fallback model")
        
        # Create some basic plots even with fallback model
        if data.valid[0].size(0) > 0:
            val_rrse, val_rae = evaluate_simple(data, data.valid[0], data.valid[1], model, 
                                 nn.MSELoss(), nn.L1Loss(), default_hp['batch_size'], True, device, "Validation")
            print("Fallback model evaluated on validation data and plots created.")
        
        # Save fallback training summary
        training_summary = {
            'hyperparameters': default_hp,
            'transfer_strategy': args.transfer_strategy,
            'pretrained_model': args.pretrained_model,
            'final_loss': float('inf'),
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.raw_data.shape,
            'device': str(device),
            'best_val_rrse': val_rrse,
            'best_val_rae': val_rae,
            'test_rrse': float('inf'),
            'test_rae': float('inf'),
            'successful_iterations': 0,
            'total_iterations': args.iterations,
            'is_fallback': True
        }
        
        with open('model/ViT_Transfer_Optim/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': default_hp,
            'is_fallback': True,
            'timestamp': datetime.now().isoformat()
        }, 'model/ViT_Transfer_Optim/best_hyperparameter_model.pt')
        
        print("Fallback model saved as 'model/ViT_Transfer_Optim/best_hyperparameter_model.pt'")

    print(f"\nHyperparameter optimization with transfer learning completed.")
    print(f"Plots saved in: model/ViT_Transfer_Optim/Validation/ and model/ViT_Transfer_Optim/Testing/")
    print(f"Model saved as: model/ViT_Transfer_Optim/best_hyperparameter_model.pt")
    print(f"Hyperparameters saved as: model/ViT_Transfer_Optim/hp.txt")
    print(f"Training summary saved as: model/ViT_Transfer_Optim/training_summary.json")
    
if __name__ == "__main__":
    main()