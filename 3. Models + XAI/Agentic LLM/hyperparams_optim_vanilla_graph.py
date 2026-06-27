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
        print(f"Loading data from: {self.data_file}")
        try:
            self.data = pd.read_csv(self.data_file, header=0)
            self.feature_names = self.data.columns.tolist()
        except Exception as e:
            print(f"Error reading with headers: {e}, trying without headers")
            self.data = pd.read_csv(self.data_file, header=None)
            self.feature_names = [f"Node_{i}" for i in range(self.data.shape[1])]
            
        self.raw_data = self.data.values
        print(f"Raw data shape: {self.raw_data.shape}")

        self.n, self.m = self.raw_data.shape
        self.col = self.feature_names

        # Create date range from July 2011 to December 2024
        from datetime import datetime, timedelta
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

def create_directories():
    directories = [
        'model/SimpleGraphTransformer/Validation',
        'model/SimpleGraphTransformer/Testing', 
        'model/SimpleGraphTransformer/plots',
        'model/SimpleGraphTransformer/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def consistent_name(name):
    """Format node names consistently for plotting"""
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
    """Save RRSE and RAE metrics to file"""
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
    with open(f'model/SimpleGraphTransformer/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')

def plot_predicted_actual(predicted, actual, title, type, dates=None):
    """Plot predicted vs actual values with dates"""
    plt.figure(figsize=(15, 6))
    
    if dates is not None and len(dates) == len(predicted):
        x = range(len(predicted))
        date_labels = [d.strftime('%b %Y') for d in dates]
        # Show every 6th label for readability
        display_indices = list(range(0, len(date_labels), max(1, len(date_labels)//12)))
        display_labels = [date_labels[i] for i in display_indices]
    else:
        x = range(1, len(predicted) + 1)
        date_labels = None
    
    plt.plot(x, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted, '--', color='purple', label='Predicted', linewidth=2)
    
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
    plt.savefig(f'model/SimpleGraphTransformer/{type}/{title_clean}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/SimpleGraphTransformer/{type}/{title_clean}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()

def calculate_rrse_rae_comprehensive(predict, test, scale):
    """Calculate comprehensive RRSE and RAE metrics"""
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

def plot_evaluation_results(predictions, targets, data, eval_type, plot_dates=None):
    """Create comprehensive evaluation plots"""
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    
    num_nodes_to_plot = min(5, data.m)
    print(f"Creating {num_nodes_to_plot} {eval_type.lower()} plots...")

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

            save_metrics_1d(torch.from_numpy(pred_curve), torch.from_numpy(target_curve), node_name, eval_type)
            
            # Create dates for the forecast horizon
            if plot_dates is not None and len(plot_dates) >= len(pred_curve):
                forecast_dates = plot_dates[:len(pred_curve)]
            else:
                forecast_dates = None
                
            plot_predicted_actual(pred_curve, target_curve, node_name, eval_type, forecast_dates)

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

def evaluate_simple_model(data, model, eval_type, batch_size, device, is_plot=False):
    """Comprehensive evaluation function with plotting"""
    model.eval()
    
    if eval_type == "Validation":
        X, Y = data.valid
        seq_dates = getattr(data, 'valid_dates', None)
    else:
        X, Y = data.test
        seq_dates = getattr(data, 'test_dates', None)

    if X.size(0) == 0:
        print(f"No {eval_type} data available")
        return float('inf'), float('inf')

    # For plotting, pick the forecast dates for the last sequence in the split (if available)
    plot_dates = seq_dates[-1] if seq_dates and len(seq_dates) > 0 else None

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
        
        rrse, rae = calculate_rrse_rae_comprehensive(all_predictions_tensor, all_targets_tensor, data.scale)
        
        print(f"{eval_type} - RRSE: {rrse:.6f}, RAE: {rae:.6f}")
        
        if is_plot and all_predictions_tensor.numel() > 0:
            plot_evaluation_results(all_predictions_tensor, all_targets_tensor, data, eval_type, plot_dates)
            
        return rrse, rae
        
    except Exception as e:
        print(f"Error in {eval_type.lower()} evaluation: {e}")
        return float('inf'), float('inf')

def main():
    parser = argparse.ArgumentParser(description='Simple Graph Transformer Hyperparameter Optimization')
    parser.add_argument('--data', type=str, default='./data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs per iteration')
    parser.add_argument('--iterations', type=int, default=20, help='number of random search iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    create_directories()
    set_random_seed(123)
    
    # Hyperparameter search space
    search_space = {
        'channels': [32, 64, 96],
        'num_layers': [2, 3, 4],
        'pe_dim': [4, 8, 12],
        'pe_walk_length': [20],
        'node_dim': [1],
        'num_heads': [4, 6, 8],
        'attn_dropout': [0.1, 0.2],
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [1e-5, 1e-4],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [8, 16, 32],
        'forecast_horizon': [36],
        'sequence_length': [12, 18, 24],
        'correlation_threshold': [0.2, 0.3, 0.4],
        'local_gnn_type': ['GINE', 'GCN'],
        'num_nodes': None  # Will be set from data
    }
    
    print("Loading data...")
    try:
        data = SimpleDataLoader(args.data, device, seq_len=18, out_len=36)
        search_space['num_nodes'] = [data.m]
        
        print(f"Data loaded successfully!")
        print(f"Training sequences: {data.train[0].shape[0]}")
        print(f"Validation sequences: {data.valid[0].shape[0]}")
        print(f"Test sequences: {data.test[0].shape[0]}")
        print(f"Number of nodes: {data.m}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Optimization tracking
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
        'search_space': search_space,
        'iterations': [],
        'best_results': None,
        'total_successful': 0,
        'total_failed': 0
    }
    
    best_val_rrse = float('inf')
    best_val_rae = float('inf')
    best_hp = None
    best_model_state = None
    
    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}")
        
        # Sample hyperparameters
        hp = {}
        for key, values in search_space.items():
            if key == 'num_nodes':
                hp[key] = data.m
            else:
                hp[key] = random.choice(values)
        
        print(f"Hyperparameters: {hp}")
        
        iteration_log = {
            'iteration': iteration + 1,
            'hyperparameters': hp,
            'status': 'failed',
            'training_loss': None,
            'validation_metrics': None
        }
        
        try:
            # Reinitialize data loader if sequence length changed
            if hp['sequence_length'] != 18:
                data = SimpleDataLoader(
                    args.data, device, 
                    seq_len=hp['sequence_length'], 
                    out_len=36,
                    correlation_threshold=hp['correlation_threshold']
                )
                hp['num_nodes'] = data.m
            
            model = SimpleGraphTransformer(hp).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
            
            train_loss = train_simple_model(data, model, optimizer, criterion, hp['batch_size'], device, args.epochs)
            iteration_log['training_loss'] = train_loss
            
            if math.isinf(train_loss):
                iteration_log['error'] = 'Training failed - infinite loss'
                optimization_log['total_failed'] += 1
                continue
                
            val_rrse, val_rae = evaluate_simple_model(data, model, "Validation", hp['batch_size'], device, False)
            iteration_log['validation_metrics'] = {'rrse': val_rrse, 'rae': val_rae}
            
            if (not math.isinf(val_rrse) and not math.isnan(val_rrse) and 
                val_rrse < best_val_rrse):
                best_val_rrse = val_rrse
                best_val_rae = val_rae
                best_hp = hp.copy()
                best_model_state = model.state_dict().copy()
                iteration_log['status'] = 'success - new best'
                optimization_log['total_successful'] += 1
                print(f"*** NEW BEST! Iteration {iteration + 1} - RRSE: {best_val_rrse:.6f}, RAE: {best_val_rae:.6f} ***")
            else:
                iteration_log['status'] = 'success'
                optimization_log['total_successful'] += 1
                print(f"Iteration {iteration + 1} completed - Val RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
                
        except Exception as e:
            iteration_log['error'] = str(e)
            optimization_log['total_failed'] += 1
            print(f"Iteration {iteration + 1} failed: {e}")
        
        optimization_log['iterations'].append(iteration_log)
    
    print("\nFINAL EVALUATION AND PLOTTING OF BEST MODEL")
    
    optimization_log['end_time'] = datetime.now().isoformat()
    
    if best_hp is not None and best_model_state is not None:
        optimization_log['best_results'] = {
            'hyperparameters': best_hp,
            'validation_rrse': best_val_rrse,
            'validation_rae': best_val_rae,
            'model_saved': True
        }
        
        # Save best hyperparameters
        with open('model/SimpleGraphTransformer/best_hyperparameters.json', 'w') as f:
            json.dump(best_hp, f, indent=2, cls=NumpyEncoder)
        
        # Create and save best model
        model = SimpleGraphTransformer(best_hp).to(device)
        model.load_state_dict(best_model_state)
        
        print("Final validation evaluation with plotting...")
        val_rrse, val_rae = evaluate_simple_model(data, model, "Validation", best_hp['batch_size'], device, True)
        
        if data.test[0].size(0) > 0:
            print("Final test evaluation with plotting...")
            test_rrse, test_rae = evaluate_simple_model(data, model, "Testing", best_hp['batch_size'], device, True)
        else:
            test_rrse, test_rae = float('inf'), float('inf')
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': best_hp,
            'validation_rrse': val_rrse,
            'validation_rae': val_rae,
            'test_rrse': test_rrse,
            'test_rae': test_rae,
            'timestamp': datetime.now().isoformat()
        }, 'model/SimpleGraphTransformer/best_model.pt')
        
        print(f"\nOPTIMIZATION COMPLETE")
        print(f"Best Validation - RRSE: {val_rrse:.6f}, RAE: {val_rae:.6f}")
        if data.test[0].size(0) > 0:
            print(f"Best Testing - RRSE: {test_rrse:.6f}, RAE: {test_rae:.6f}")
        
    else:
        print("No successful iterations completed.")
        # Create fallback model
        default_hp = {
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
            'forecast_horizon': 36,
            'sequence_length': 18,
            'correlation_threshold': 0.3,
            'local_gnn_type': 'GINE',
            'num_nodes': data.m
        }
        
        with open('model/SimpleGraphTransformer/best_hyperparameters.json', 'w') as f:
            json.dump(default_hp, f, indent=2, cls=NumpyEncoder)
        
        model = SimpleGraphTransformer(default_hp).to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': default_hp,
            'is_fallback': True,
            'timestamp': datetime.now().isoformat()
        }, 'model/SimpleGraphTransformer/best_model.pt')
        
        print("Fallback model saved")
    
    # Save optimization log
    with open('model/SimpleGraphTransformer/optimization_log.json', 'w') as f:
        json.dump(optimization_log, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nHyperparameter optimization completed.")
    print(f"Successful iterations: {optimization_log['total_successful']}")
    print(f"Failed iterations: {optimization_log['total_failed']}")
    print(f"Detailed log saved: model/SimpleGraphTransformer/optimization_log.json")
    print(f"Plots saved in: model/SimpleGraphTransformer/Validation/ and model/SimpleGraphTransformer/Testing/")
    print(f"Model saved as: model/SimpleGraphTransformer/best_model.pt")
    print(f"Hyperparameters saved as: model/SimpleGraphTransformer/best_hyperparameters.json")
    
if __name__ == "__main__":
    main()