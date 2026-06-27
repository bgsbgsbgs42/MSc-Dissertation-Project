import ast
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

        class LocalGNNWrapper(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv
            def forward(self, x, edge_index, edge_attr=None):
                if isinstance(self.conv, GINEConv):
                    return self.conv(x, edge_index, edge_attr)
                else:
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
        seq_dates = []
        for i in range(n_seq):
            X_all[i] = self.normalized_data[i:i+self.seq_len]
            Y_all[i] = self.normalized_data[i+self.seq_len:i+self.seq_len+self.out_len]
            start_idx = i + self.seq_len
            seq_dates.append(self.dates[start_idx:start_idx + self.out_len])

        # Split sequences into train/valid/test by ratio
        n_train = int(n_seq * 0.7)
        n_valid = int(n_seq * 0.15)
        n_test  = n_seq - n_train - n_valid

        def to_tensors(arr):
            return torch.from_numpy(arr).float()

        self.train = (to_tensors(X_all[:n_train]), to_tensors(Y_all[:n_train]))
        self.valid = (to_tensors(X_all[n_train:n_train + n_valid]), to_tensors(Y_all[n_train:n_train + n_valid]))
        self.test  = (to_tensors(X_all[n_train + n_valid:]), to_tensors(Y_all[n_train + n_valid:]))

        self.train_dates = seq_dates[:n_train]
        self.valid_dates = seq_dates[n_train:n_train + n_valid]
        self.test_dates  = seq_dates[n_train + n_valid:]

        print(f"Train sequences: {self.train[0].shape[0]}")
        print(f"Validation sequences: {self.valid[0].shape[0]}")
        print(f"Test sequences: {self.test[0].shape[0]}")
    
    def prepare_pyg_data(self, sequences, targets, walk_length=20):
        """Convert sequences to PyG Data objects"""
        data_list = []
        
        for seq, target in zip(sequences, targets):
            x = torch.FloatTensor(seq[-1])
            
            transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
            
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

    def get_batches(self, X, Y, batch_size, shuffle=True):
        """Generator for batches - similar to original script"""
        n_samples = X.size(0)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], Y[batch_indices]

def train_model(data, model, criterion, optimizer, batch_size, device, epochs):
    """Training function matching the original script's structure"""
    model.train()
    
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        epoch_start_time = time.time()
        
        total_loss = 0
        n_samples = 0
        iter = 0
        
        # Use the batch generator approach from original script
        for X, Y in data.get_batches(data.train[0], data.train[1], batch_size, True):
            model.zero_grad()
            
            # Prepare PyG data for this batch
            batch_data = data.prepare_pyg_data(X.numpy(), Y.numpy())
            if not batch_data:
                continue
                
            batch_loader = PyGDataLoader(batch_data, batch_size=len(batch_data), shuffle=False)
            
            for batch in batch_loader:
                batch = batch.to(device)
                
                output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Reshape target to match output
                batch_size_out = output.shape[0]
                target_reshaped = batch.y.view(batch_size_out, data.out_len, data.m)
                
                # Apply scaling for loss calculation (similar to original)
                scale_batch = data.scale.expand(output.size(0), output.size(1), data.m).to(device)
                output_scaled = output * scale_batch
                target_scaled = target_reshaped * scale_batch
                
                loss = criterion(output_scaled, target_scaled)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_samples += (output.size(0) * output.size(1) * data.m)
            
            if iter % 1 == 0:
                current_loss = loss.item() / (output.size(0) * output.size(1) * data.m)
                print(f'iter:{iter:3d} | loss: {current_loss:.3f}')
            iter += 1
        
        if n_samples > 0:
            avg_loss = total_loss / n_samples
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch} completed | Average Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s')
        else:
            print(f'Epoch {epoch} - No training data')

def create_directories():
    """Create necessary directories"""
    directories = [
        'model/SimpleGraphTransformer',
        'model/SimpleGraphTransformer/plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='Train Simple Graph Transformer with optimal hyperparameters')
    parser.add_argument('--data', type=str, default='Dissertation/Vanilla Graph Transformer/data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--save', type=str, default='model/SimpleGraphTransformer/final_model.pt', help='path to save the final model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Set random seed for reproducibility
    fixed_seed = 123
    set_random_seed(fixed_seed)
    
    create_directories()
    
    # Load optimal hyperparameters
    hp_file = "model/SimpleGraphTransformer/best_hyperparameters.json"
    try:
        with open(hp_file, 'r') as f:
            hp = json.load(f)
        print('Loaded optimal hyperparameters:', hp)
    except FileNotFoundError:
        print(f"Warning: {hp_file} not found. Using default hyperparameters.")
        # Default fallback hyperparameters
        hp = {
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
            'local_gnn_type': 'GINE'
        }
    
    # Load data
    print('Loading data...')
    try:
        data = SimpleDataLoader(
            args.data, 
            device, 
            seq_len=hp['sequence_length'], 
            out_len=hp['forecast_horizon'],
            correlation_threshold=hp['correlation_threshold']
        )
        hp['num_nodes'] = data.m
        
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
    
    # Create model
    print('Creating model...')
    model = SimpleGraphTransformer(hp).to(device)
    
    # Print model info
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    print('Model architecture:', model)
    
    # Define loss and optimizer
    criterion = nn.L1Loss(reduction='sum').to(device)  
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=hp['learning_rate'], 
        weight_decay=hp['weight_decay']
    )
    
    print('Beginning training...')
    print(f'Training on: {device}')
    print(f'Number of epochs: {args.epochs}')
    print(f'Batch size: {hp["batch_size"]}')
    print(f'Learning rate: {hp["learning_rate"]}')
    
    # Train the model
    try:
        train_model(data, model, criterion, optimizer, hp['batch_size'], device, args.epochs)
        
        # Save the final model
        torch.save(model.state_dict(), args.save)
        print(f'Model saved to: {args.save}')
        
        # Also save model info
        # Create a JSON-serializable summary of the model state dict (shapes and dtypes)
        state_summary = {
            k: {'shape': list(v.size()), 'dtype': str(v.dtype)}
            for k, v in model.state_dict().items()
        }

        model_info = {
            'hyperparameters': hp,
            'model_state_dict_summary': state_summary,
            'model_state_dict_saved_path': args.save,  # actual tensors saved as binary with torch.save above
            'training_completed': True,
            'epochs': args.epochs,
            'timestamp': datetime.now().isoformat(),
            'parameters': nParams,
            'data_info': {
                'file': args.data,
                'nodes': data.m,
                'training_sequences': data.train[0].shape[0],
                'validation_sequences': data.valid[0].shape[0],
                'test_sequences': data.test[0].shape[0]
            }
        }
        
        info_file = args.save.replace('.pt', '_info.json')
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f'Model info saved to: {info_file}')
        
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        # Save model even if interrupted
        torch.save(model.state_dict(), args.save)
        print(f'Model saved to: {args.save} (early termination)')
    except Exception as e:
        print(f'Training error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()