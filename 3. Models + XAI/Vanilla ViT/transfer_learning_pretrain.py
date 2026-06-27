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
    """Simple patch embedding for time series"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Ensure sequence length is divisible by patch size
        if seq_len % patch_size != 0:
            divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
            if divisors:
                patch_size = max(divisors)
                print(f"Adjusted patch_size to: {patch_size} (divisor of {seq_len})")
            else:
                patch_size = 1
                print(f"Using fallback patch_size: 1")
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        print(f"Final: seq_len={seq_len}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Simple linear projection for patches
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes)
        batch_size = x.shape[0]
        
        # Reshape to patches
        x = x.unfold(1, self.patch_size, self.patch_size)  # (batch_size, num_patches, num_nodes, patch_size)
        x = x.contiguous().view(batch_size, self.num_patches, -1)  # (batch_size, num_patches, num_nodes * patch_size)
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch_size, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        return x

class SimpleVisionTransformer(nn.Module):
    """Simplified Vision Transformer for time series forecasting"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self._validate_config(config)
        
        # Calculate actual patch size
        actual_patch_size = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
        actual_num_patches = config['sequence_length'] // actual_patch_size
        
        print(f"Model config: sequence_length={config['sequence_length']}, patch_size={actual_patch_size}, num_patches={actual_num_patches}")
        
        self.patch_embed = SimplePatchEmbedding(
            seq_len=config['sequence_length'],
            patch_size=actual_patch_size,
            in_channels=config['num_nodes'],
            embed_dim=config['embed_dim']
        )
        
        # Simple transformer encoder
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
        
        # Simple forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(config['embed_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['forecast_horizon'] * config['num_nodes'])
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
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
        
        # Ensure compatibility
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
            return 1
        
        return max(divisors)
    
    def _get_compatible_heads(self, embed_dim, desired_heads):
        """Find compatible number of heads that divides embed_dim"""
        divisors = []
        for i in range(1, embed_dim + 1):
            if embed_dim % i == 0:
                divisors.append(i)
        
        if not divisors:
            return 1
        
        compatible_heads = [h for h in divisors if h <= min(desired_heads, 16)]
        if compatible_heads:
            return max(compatible_heads)
        else:
            return min(divisors)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes)
        x = self.patch_embed(x)
        x = self.transformer(x)
        
        # Use CLS token for forecasting
        cls_token = x[:, 0]
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
        """Split data into train, validation, and test sets"""
        total_samples = len(self.normalized_data)
        
        n_train = int(total_samples * self.train_ratio)
        n_valid = int(total_samples * self.valid_ratio)
        
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {n_train}, Validation samples: {n_valid}")
        
        train_data = self.normalized_data[:n_train]
        valid_data = self.normalized_data[n_train:n_train + n_valid]
        test_data = self.normalized_data[n_train + n_valid:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Create sequences
        self.train = self._batchify(train_data, self.seq_len, self.out_len)
        self.valid = self._batchify(valid_data, self.seq_len, self.out_len)
        self.test = self._batchify(test_data, self.seq_len, self.out_len)
        
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

def create_directories():
    """Create all necessary directories for transfer learning"""
    directories = [
        'transfer_learning_pretrain',
        'transfer_learning_pretrain/pretrained_models',
        'transfer_learning_pretrain/plots',
        'transfer_learning_pretrain/results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def train_simple(data, X, Y, model, criterion, optimizer, batch_size, device):
    """Simple training function"""
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
        
        # Simple gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        total_loss += loss.item()
        n_samples += (output.size(0) * output.size(1) * data.m)
        batch_count += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / n_samples if n_samples > 0 else total_loss
    print(f"  Training completed: {batch_count} batches, Avg Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_simple(data, X, Y, model, evaluateL2, evaluateL1, batch_size, device, eval_type="Validation"):
    """Simple evaluation function"""
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0

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
            
            # Calculate losses
            total_loss += evaluateL2(output, Y_batch[:, :min_horizon, :]).item()
            total_loss_l1 += evaluateL1(output, Y_batch[:, :min_horizon, :]).item()
            n_samples += (output.size(0) * output.size(1) * data.m)

    if n_samples == 0:
        print(f"CRITICAL: No samples were processed during {eval_type.lower()} evaluation")
        return float('inf'), float('inf')

    avg_loss = total_loss / n_samples
    avg_loss_l1 = total_loss_l1 / n_samples
    
    print(f"{eval_type} evaluation completed - MSE: {avg_loss:.6f}, MAE: {avg_loss_l1:.6f}")
    
    return avg_loss, avg_loss_l1

class PreTrainManager:
    """Manages pre-training on multiple domains"""
    
    def __init__(self, base_config, device):
        self.base_config = base_config
        self.device = device
        self.pretrained_models = []
    
    def load_domain_data(self, csv_file, domain_name):
        """Load and prepare domain data for pre-training"""
        print(f"Loading domain data: {domain_name}")
        
        try:
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_cols].values
            
            print(f"  Data shape: {data.shape}")
            print(f"  Features: {len(numeric_cols)}")
            
            # Normalize data
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std[data_std == 0] = 1.0
            data_normalized = (data - data_mean) / data_std
            
            # Create dataset
            dataset = DomainDataset(data_normalized, 
                                  sequence_length=self.base_config['sequence_length'],
                                  forecast_horizon=6)  # Shorter horizon for pre-training
            
            if len(dataset) == 0:
                print(f"  WARNING: No sequences could be created for {domain_name}")
                return None
            
            # Split into train/validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(dataset)))
            
            X_train = torch.stack([dataset[i][0] for i in train_indices])
            y_train = torch.stack([dataset[i][1] for i in train_indices])
            X_val = torch.stack([dataset[i][0] for i in val_indices])
            y_val = torch.stack([dataset[i][1] for i in val_indices])
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'num_nodes': data.shape[1],
                'domain_name': domain_name
            }
            
        except Exception as e:
            print(f"Error loading domain data {domain_name}: {e}")
            return None
    
    def pretrain_domain(self, domain_data, epochs=50):
        """Pre-train model on a single domain"""
        if domain_data is None:
            return None
            
        domain_name = domain_data['domain_name']
        print(f"\nPre-training on domain: {domain_name}")
        
        # Adjust config for this domain
        config = self.base_config.copy()
        config['num_nodes'] = domain_data['num_nodes']
        config['forecast_horizon'] = domain_data['y_train'].shape[1]
        
        # Create model
        model = SimpleVisionTransformer(config).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0
            n_batches = 0
            
            indices = torch.randperm(len(domain_data['X_train']))
            batch_size = min(config['batch_size'], len(domain_data['X_train']))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = domain_data['X_train'][batch_indices].to(self.device)
                y_batch = domain_data['y_train'][batch_indices].to(self.device)
                
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
                val_output = model(domain_data['X_val'].to(self.device))
                val_loss = criterion(val_output, domain_data['y_val'].to(self.device))
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict().copy()
        
        print(f"Pre-training completed for {domain_name}. Best val loss: {best_val_loss:.6f}")
        
        return {
            'domain_name': domain_name,
            'model_state': best_model_state,
            'val_loss': best_val_loss,
            'config': config
        }

class TransferLearningManager:
    """Manages transfer learning from pre-trained models to cyber threat data"""
    
    def __init__(self, pretrained_models, device):
        self.pretrained_models = pretrained_models
        self.device = device
    
    def transfer_weights(self, target_model, strategy='best'):
        """Transfer weights from pre-trained models to target model"""
        if not self.pretrained_models:
            print("No pre-trained models available for transfer")
            return target_model, []
        
        if strategy == 'best':
            # Use the best pre-trained model
            best_model = min(self.pretrained_models, key=lambda x: x['val_loss'])
            return self._transfer_single_model(target_model, best_model)
        elif strategy == 'ensemble':
            # Average weights from multiple models
            return self._transfer_ensemble(target_model)
        else:
            # Use first available model
            return self._transfer_single_model(target_model, self.pretrained_models[0])
    
    def _transfer_single_model(self, target_model, source_model_info):
        """Transfer weights from a single pre-trained model"""
        source_state = source_model_info['model_state']
        target_state = target_model.state_dict()
        
        transferred_layers = []
        
        for name, param in source_state.items():
            if name in target_state:
                if target_state[name].shape == param.shape:
                    target_state[name] = param
                    transferred_layers.append(name)
                else:
                    print(f"  Shape mismatch for {name}: source {param.shape}, target {target_state[name].shape}")
        
        target_model.load_state_dict(target_state)
        print(f"Transferred {len(transferred_layers)} layers from {source_model_info['domain_name']}")
        
        return target_model, transferred_layers
    
    def _transfer_ensemble(self, target_model):
        """Average weights from multiple pre-trained models"""
        print("Averaging weights from multiple pre-trained models...")
        
        target_state = target_model.state_dict()
        weight_counts = {}
        
        for pretrained in self.pretrained_models:
            source_state = pretrained['model_state']
            
            for name, param in source_state.items():
                if name in target_state and target_state[name].shape == param.shape:
                    if name not in weight_counts:
                        weight_counts[name] = []
                    weight_counts[name].append(param.cpu())
        
        # Average weights
        transferred_layers = []
        for name, weights in weight_counts.items():
            if len(weights) > 0:
                avg_weight = torch.stack(weights).mean(dim=0)
                target_state[name] = avg_weight
                transferred_layers.append(name)
        
        target_model.load_state_dict(target_state)
        print(f"Averaged and transferred {len(transferred_layers)} layers from {len(self.pretrained_models)} domains")
        
        return target_model, transferred_layers

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Pre-training and Fine-tuning')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=50, help='epochs for pre-training each domain')
    parser.add_argument('--epochs_finetune', type=int, default=20, help='epochs for fine-tuning on cyber data')
    parser.add_argument('--transfer_strategy', type=str, default='best', choices=['best', 'ensemble', 'first'], help='strategy for weight transfer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create directories
    create_directories()
    
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
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 8,
        'forecast_horizon': 12,
    }
    
    
    # PRE-TRAINING ON RELATED DOMAINS
    
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON RELATED DOMAINS")
    print("="*80)
    
    # Find all CSV files in pretrain directory
    pretrain_files = glob.glob(os.path.join(args.pretrain_data_dir, "*.csv"))
    print(f"Found {len(pretrain_files)} pre-training files")
    
    pretrain_manager = PreTrainManager(base_config, device)
    pretrained_models = []
    
    for file_path in pretrain_files:
        domain_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load domain data
        domain_data = pretrain_manager.load_domain_data(file_path, domain_name)
        
        if domain_data is not None:
            # Pre-train on this domain
            pretrained_model = pretrain_manager.pretrain_domain(
                domain_data, epochs=args.epochs_pretrain
            )
            
            if pretrained_model is not None:
                pretrained_models.append(pretrained_model)
                
                # Save pre-trained model
                model_path = f'transfer_learning_pretrain/pretrained_models/pretrained_{domain_name}.pt'
                torch.save(pretrained_model, model_path)
                print(f"Saved pre-trained model: {model_path}")
    
    print(f"\nPre-training completed. {len(pretrained_models)} models trained successfully.")
    
    
    #  TRANSFER & INITIAL FINE-TUNING
    
    print("\n" + "="*80)
    print("PHASE 2: TRANSFER & INITIAL FINE-TUNING ON CYBER THREAT DATA")
    print("="*80)
    
    # Load cyber threat data
    print("Loading cyber threat data...")
    try:
        cyber_data = SimpleDataLoader(args.cyber_data, 0.6, 0.2, device, horizon=1, seq_len=12, normalize=2, out_len=12)
        print(f"Cyber threat data loaded successfully!")
        print(f"Number of nodes: {cyber_data.m}")
        
    except Exception as e:
        print(f"Error loading cyber threat data: {e}")
        return
    
    # Create target model configuration for cyber threat data
    cyber_config = base_config.copy()
    cyber_config['num_nodes'] = cyber_data.m
    cyber_config['forecast_horizon'] = cyber_data.train[1].shape[1]
    
    # Create target model
    target_model = SimpleVisionTransformer(cyber_config).to(device)
    
    if pretrained_models:
        # Transfer weights from pre-trained models
        print(f"\nTransferring weights using strategy: {args.transfer_strategy}")
        transfer_manager = TransferLearningManager(pretrained_models, device)
        transferred_model, transferred_layers = transfer_manager.transfer_weights(
            target_model, strategy=args.transfer_strategy
        )
        
        print(f"Successfully transferred {len(transferred_layers)} layers")
    else:
        print("No pre-trained models available, using random initialization")
        transferred_model = target_model
        transferred_layers = []
    
    # Fine-tuning on cyber threat data
    print(f"\nStarting fine-tuning on cyber threat data for {args.epochs_finetune} epochs...")
    
    criterion = nn.MSELoss()
    evaluateL2 = nn.MSELoss()
    evaluateL1 = nn.L1Loss()
    
    # Use lower learning rate for fine-tuning
    optimizer = optim.AdamW(transferred_model.parameters(), lr=0.0001, weight_decay=0.001)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, args.epochs_finetune + 1):
        # Training
        train_loss = train_simple(cyber_data, cyber_data.train[0], cyber_data.train[1], 
                                transferred_model, criterion, optimizer, 
                                cyber_config['batch_size'], device)
        
        # Validation
        val_loss, val_mae = evaluate_simple(cyber_data, cyber_data.valid[0], cyber_data.valid[1],
                                          transferred_model, evaluateL2, evaluateL1,
                                          cyber_config['batch_size'], device, "Validation")
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = transferred_model.state_dict().copy()
            print(f"  -> New best validation loss: {best_val_loss:.6f}")
    
    # Save the fine-tuned model
    model_save_data = {
        'model_state_dict': best_model_state,
        'hyperparameters': cyber_config,
        'transferred_layers': transferred_layers,
        'pretrained_domains': [model['domain_name'] for model in pretrained_models],
        'transfer_strategy': args.transfer_strategy,
        'final_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(model_save_data, 'transfer_learning_pretrain/transferred_model_finetuned.pt')
    
    # Test evaluation
    print("\nFinal evaluation on test set...")
    test_loss, test_mae = evaluate_simple(cyber_data, cyber_data.test[0], cyber_data.test[1],
                                        transferred_model, evaluateL2, evaluateL1,
                                        cyber_config['batch_size'], device, "Testing")
    
    # Save training summary
    training_summary = {
        'pretrained_domains': [model['domain_name'] for model in pretrained_models],
        'transfer_strategy': args.transfer_strategy,
        'epochs_pretrain': args.epochs_pretrain,
        'epochs_finetune': args.epochs_finetune,
        'final_val_loss': best_val_loss,
        'final_test_loss': test_loss,
        'final_test_mae': test_mae,
        'transferred_layers_count': len(transferred_layers),
        'cyber_data_shape': cyber_data.raw_data.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('transfer_learning_pretrain/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nTransfer learning pipeline completed successfully!")
    print(f"Final validation loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test MAE: {test_mae:.6f}")
    print(f"Transferred layers: {len(transferred_layers)}")
    print(f"Results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()