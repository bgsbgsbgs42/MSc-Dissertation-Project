import ast
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

def train(data, X, Y, model, criterion, optimizer, batch_size, device):
    """Training function for Vanilla ViT"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    iter = 0

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
        
        if iter % 1 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item()/(output.size(0) * output.size(1) * data.m)))
        iter += 1

    if batch_count == 0:
        print("CRITICAL: No valid batches processed during training")
        return float('inf')

    avg_loss = total_loss / batch_count
    print(f"Training completed: {batch_count} batches, Avg Loss: {avg_loss:.6f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Vanilla ViT Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv',
                        help='location of the data file')
    parser.add_argument('--save', type=str, default='model/ViT_Final/o_model.pt',
                        help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='device')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Set random seed
    set_random_seed(123)
    
    print("Loading data...")
    try:
        # Read hyperparameters from the optimization result
        filename = "model/ViT_Transfer_Optim/hp.txt"
        with open(filename, 'r') as file:
            content = file.read()
            hp = json.loads(content)
        
        print('Loaded hyperparameters:', hp)
        
        # Create data loader with optimal hyperparameters
        data = SimpleDataLoader(args.data, 0.6, 0.2, device, horizon=1, 
                               seq_len=hp['sequence_length'], normalize=args.normalize, 
                               out_len=hp['forecast_horizon'])
        print(f"Data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data or hyperparameters: {e}")
        # Fallback to default parameters
        print("Using default parameters as fallback")
        hp = {
            'sequence_length': 24,
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
            'num_nodes': data.m if 'data' in locals() else 142
        }
        data = SimpleDataLoader(args.data, 0.6, 0.2, device, horizon=1, 
                               seq_len=hp['sequence_length'], normalize=args.normalize, 
                               out_len=hp['forecast_horizon'])
    
    # Update num_nodes in hyperparameters
    hp['num_nodes'] = data.m
    
    print("Initializing model...")
    model = SimpleVisionTransformer(hp).to(device)
    
    print("Model configuration:")
    print(f"  Sequence length: {hp['sequence_length']}")
    print(f"  Patch size: {hp['patch_size']}")
    print(f"  Embed dim: {hp['embed_dim']}")
    print(f"  Num heads: {hp['num_heads']}")
    print(f"  Hidden dim: {hp['hidden_dim']}")
    print(f"  Num layers: {hp['num_layers']}")
    print(f"  Dropout: {hp['dropout']}")
    print(f"  Forecast horizon: {hp['forecast_horizon']}")
    print(f"  Num nodes: {hp['num_nodes']}")
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    
    # Setup loss functions and optimizer
    if args.L1Loss:
        criterion = nn.L1Loss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='sum').to(device)
    
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)  # MSE
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)   # MAE
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'], 
                              weight_decay=hp['weight_decay'])
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], 
                               weight_decay=hp['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'], 
                              weight_decay=hp['weight_decay'])
    
    print(f"Training on {len(data.train[0])} sequences")
    print(f"Optimizer: {args.optim}, Learning rate: {hp['learning_rate']}")
    print(f"Batch size: {args.batch_size}")
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    
    # Training loop
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            print('epoch:', epoch)
            epoch_start_time = time.time()
            train_loss = train(data, data.train[0], data.train[1], model, criterion, 
                              optimizer, args.batch_size, device)
            epoch_end_time = time.time()
            print('epoch time: {:.2f}s'.format(epoch_end_time - epoch_start_time))
        
        # Save the trained model
        model_save_data = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': hp,
            'training_config': vars(args),
            'timestamp': datetime.now().isoformat()
        }
        torch.save(model_save_data, args.save)
        print(f"Model saved to {args.save}")
        
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        # Save model even if training is interrupted
        model_save_data = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': hp,
            'training_config': vars(args),
            'timestamp': datetime.now().isoformat(),
            'interrupted': True
        }
        torch.save(model_save_data, args.save)
        print(f"Model saved to {args.save} (interrupted)")

if __name__ == "__main__":
    main()