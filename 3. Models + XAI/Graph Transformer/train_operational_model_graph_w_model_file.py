import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 1200

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

class RedrawProjection:
    def __init__(self, model, redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pred_len = config['forecast_horizon']
        self.channels = config['channels']
        self.num_nodes = config['num_nodes']
        
        self.node_emb = nn.Linear(config['node_dim'], config['channels'] - config['pe_dim'])
        self.pe_lin = nn.Linear(20, config['pe_dim'])
        self.pe_norm = nn.BatchNorm1d(20)
        
        self.edge_emb = nn.Embedding(2, config['channels'])
        
        self.convs = nn.ModuleList()
        for _ in range(config['num_layers']):
            nn_seq = nn.Sequential(
                nn.Linear(config['channels'], config['channels']),
                nn.ReLU(),
                nn.Linear(config['channels'], config['channels']),
            )
            conv = GPSConv(config['channels'], GINEConv(nn_seq), heads=config['num_heads'],
                          attn_type=config['attn_type'], attn_kwargs={'dropout': config['dropout']})
            self.convs.append(conv)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.forecast_head = nn.Sequential(
            nn.Linear(config['channels'], config['channels'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['channels'] // 2, config['channels'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['channels'] // 4, config['forecast_horizon'] * config['num_nodes']),
        )
        
        self.redraw_projection = RedrawProjection(
            self, redraw_interval=1000 if config['attn_type'] == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch, mc_dropout=True):
        x_pe = self.pe_norm(pe)
        
        node_emb = self.node_emb(x)
        pe_emb = self.pe_lin(x_pe)
        
        x = torch.cat((node_emb, pe_emb), dim=1)
        edge_attr = self.edge_emb(edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            if mc_dropout:
                x = self.dropout(x)
        
        x = global_add_pool(x, batch)
        return self.forecast_head(x)

class OperationalDataLoader:
    def __init__(self, data_file, device, seq_len=12, out_len=36):
        self.data_file = data_file
        self.device = device
        self.seq_len = seq_len
        self.out_len = out_len
        
        self.load_full_data()
        self.create_graph_structure()
        
    def load_full_data(self):
        print(f"Loading full operational data from: {self.data_file}")
        # Try to infer the delimiter first (handles csv or tab-separated files)
        try:
            df = pd.read_csv(self.data_file, sep=None, engine='python')
        except Exception:
            # Fallback to tab-separated with no header
            df = pd.read_csv(self.data_file, delimiter='\t', header=None)

        # Coerce all columns to numeric; non-numeric -> NaN
        numeric_df = df.apply(pd.to_numeric, errors='coerce')

        # If coercion produced all-NaN (e.g., header row present), try reading with header=0 then coerce
        if numeric_df.isnull().all().all():
            try:
                df_header = pd.read_csv(self.data_file, sep=None, engine='python', header=0)
                numeric_df = df_header.apply(pd.to_numeric, errors='coerce')
            except Exception:
                # As a last resort, coerce original df and continue
                numeric_df = df.apply(pd.to_numeric, errors='coerce')

        # Fill NaNs produced by coercion using forward/backward fill then remaining with zeros
        numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

        # Convert to numpy and transpose to match original shape expectations (time x nodes)
        self.data = numeric_df
        self.raw_data = numeric_df.values.T.astype(float)
        print(f"Full data shape (time_steps, nodes): {self.raw_data.shape}")

        self.n, self.m = self.raw_data.shape
        # Preserve column names if available; otherwise create Node_<i>
        try:
            cols = list(numeric_df.columns)
            if all(isinstance(c, str) for c in cols):
                self.col = cols
            else:
                self.col = [f"Node_{i}" for i in range(self.m)]
        except Exception:
            self.col = [f"Node_{i}" for i in range(self.m)]

        # Compute scale safely on numeric data
        scale_arr = np.max(np.abs(self.raw_data), axis=0)
        scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
        self.scale = torch.from_numpy(scale_arr).float()

        # Normalize using numpy arrays (avoid torch/CPU mismatch)
        self.normalized_data = (self.raw_data / scale_arr)
        print(f"Normalized full data shape: {self.normalized_data.shape}")
        
    def create_graph_structure(self):
        correlation_matrix = np.corrcoef(self.normalized_data.T)
        threshold = 0.3
        adj_matrix = (correlation_matrix > threshold).astype(int)
        
        edge_index = []
        edge_attr = []
        for i in range(self.m):
            for j in range(self.m):
                if adj_matrix[i, j] == 1 and i != j:
                    edge_index.append([i, j])
                    edge_attr.append(1)
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        print(f"Operational graph created: {self.edge_index.shape[1]} edges")
        
    def prepare_training_data(self):
        sequences = []
        targets = []
        
        for i in range(len(self.normalized_data) - self.seq_len - self.out_len + 1):
            seq = self.normalized_data[i:i + self.seq_len]
            target = self.normalized_data[i + self.seq_len:i + self.seq_len + self.out_len]
            sequences.append(seq)
            targets.append(target)
            
        print(f"Created {len(sequences)} training sequences")
        return sequences, targets
    
    def prepare_pyg_data(self, sequences, targets):
        data_list = []
        
        for seq, target in zip(sequences, targets):
            x = torch.FloatTensor(seq[-1])
            
            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
            data = Data(
                x=x.unsqueeze(1),
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=torch.FloatTensor(target.T).flatten(),
                num_nodes=self.m
            )
            data = transform(data)
            data_list.append(data)
        
        return data_list
    
    def get_final_sequence(self):
        """Get the final sequence for forecasting future 36 months"""
        final_seq = self.normalized_data[-self.seq_len:]
        return final_seq

def load_hyperparameters(hp_file):
    with open(hp_file, 'r') as f:
        hp = json.load(f)
    print("Loaded hyperparameters:", hp)
    return hp

def safe_load_pretrained(model, state_dict, device):
    """
    Copy matching parameters from state_dict into model.
    - Exact-shape copy when possible.
    - Partial copy for overlapping slices when shapes differ but ranks match.
    - Ignores unexpected keys in checkpoint.
    Returns (transferred_list, skipped_list).
    """
    model_state = model.state_dict()
    transferred = []
    skipped = []

    for name, param in model_state.items():
        if name not in state_dict:
            skipped.append((name, 'missing_in_checkpoint'))
            continue

        ckpt_param = state_dict[name]
        # Ensure ckpt_param is a tensor
        if not isinstance(ckpt_param, torch.Tensor):
            try:
                ckpt_param = torch.tensor(ckpt_param)
            except Exception:
                skipped.append((name, 'not_tensor'))
                continue

        # Exact shape match -> copy
        if tuple(ckpt_param.shape) == tuple(param.shape):
            try:
                param.copy_(ckpt_param.to(param.device))
                transferred.append(name)
                continue
            except Exception as e:
                skipped.append((name, f'copy_error:{e}'))
                continue

        # Partial copy when dims match (copy overlapping region)
        if ckpt_param.ndim == param.ndim and ckpt_param.ndim >= 1:
            try:
                slices = tuple(slice(0, min(s_ck, s_mod)) for s_ck, s_mod in zip(ckpt_param.shape, param.shape))
                param_view = param.__getitem__(slices)
                ckpt_view = ckpt_param.__getitem__(slices).to(param.device)
                if ckpt_view.dtype != param_view.dtype:
                    ckpt_view = ckpt_view.to(param_view.dtype)
                param_view.copy_(ckpt_view)
                transferred.append(name + '_partial')
                continue
            except Exception as e:
                skipped.append((name, f'partial_copy_error:{e}'))
                continue

        # Otherwise skip
        skipped.append((name, f'shape_mismatch_ckpt{tuple(ckpt_param.shape)}_model{tuple(param.shape)}'))

    # Optionally report a few skipped entries
    print(f"[safe_load_pretrained] transferred {len(transferred)} params, skipped {len(skipped)} params")
    if skipped:
        for s in skipped[:20]:
            print(f"  SKIPPED: {s[0]} -> {s[1]}")
    return transferred, skipped

def load_pretrained_model(model_path, device, data):
    """Load a pre-trained model and its configuration with robust fallback loading."""
    print(f"Loading pre-trained model from: {model_path}")
    model_data = torch.load(model_path, map_location=device, weights_only=False)

    # Extract hyperparameters and state dict (support different checkpoint layouts)
    if isinstance(model_data, dict) and 'model_state_dict' in model_data:
        state_dict = model_data['model_state_dict']
    elif isinstance(model_data, dict) and 'state_dict' in model_data:
        state_dict = model_data['state_dict']
    else:
        # Assume entire object is state_dict
        state_dict = model_data

    # Extract hyperparameters if present
    hp = {}
    if isinstance(model_data, dict) and 'hyperparameters' in model_data:
        hp = model_data['hyperparameters']
    elif isinstance(model_data, dict) and 'config' in model_data:
        hp = model_data['config']

    # Ensure num_nodes matches current data
    if isinstance(hp, dict):
        hp['num_nodes'] = data.m
    else:
        hp = {'num_nodes': data.m, 'channels': 64, 'forecast_horizon': data.out_len, 'num_layers': 3, 'pe_dim': 8, 'node_dim': 1, 'attn_type': 'multihead', 'dropout': 0.2}

    # Instantiate model with (possibly) updated hp
    model = GraphTransformer(hp).to(device)

    # Try strict load first for best-case (will raise on mismatches)
    try:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully (strict). All compatible weights transferred.")
    except Exception as e:
        print(f"Strict load_state_dict failed: {e}")
        print("Falling back to safe_load_pretrained to copy matching parameters and partial overlaps.")
        transferred, skipped = safe_load_pretrained(model, state_dict, device)
        print(f"safe_load_pretrained transferred {len(transferred)} params; skipped {len(skipped)} params.")

    print(f"Pre-trained model loaded. Previous training loss: {model_data.get('training_loss', model_data.get('loss', 'N/A')) if isinstance(model_data, dict) else 'N/A'}")
    return model, hp, model_data

def train_operational_model(data, model, optimizer, criterion, batch_size, device, epochs):
    model.train()
    
    sequences, targets = data.prepare_training_data()
    train_data = data.prepare_pyg_data(sequences, targets)
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            model.redraw_projection.redraw_projections()
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, mc_dropout=True)
            
            target_reshaped = batch.y.view(output.shape[0], -1)
            loss = criterion(output, target_reshaped)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        
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

def main():
    parser = argparse.ArgumentParser(description='Graph Transformer Operational Model Training (Continue Training)')
    parser.add_argument('--data', type=str, default='Dissertation/Graph Transformer/data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--hp_file', type=str, default='Dissertation/Graph Transformer/model/GraphTransformer/hp.txt', help='hyperparameter file')
    parser.add_argument('--model_path', type=str, default='Dissertation/Graph Transformer/model/GraphTransformer/best_hyperparameter_model_pretrained.pt', help='path to pre-trained model')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--L1Loss', type=bool, default=True)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(" GRAPH TRANSFORMER OPERATIONAL MODEL CONTINUED TRAINING ")
    
    print("Loading full operational data...")
    data = OperationalDataLoader(args.data, device)
    
    print(f"Training on full dataset with {data.n} time steps and {data.m} nodes")
    
    # Load pre-trained model
    try:
        model, hp, model_data = load_pretrained_model(args.model_path, device, data)
        print("Successfully loaded pre-trained model. Continuing training...")
    except FileNotFoundError:
        print(f"Pre-trained model {args.model_path} not found. Training from scratch...")
        try:
            hp = load_hyperparameters(args.hp_file)
        except FileNotFoundError:
            print(f"Hyperparameter file {args.hp_file} not found. Using default parameters.")
            hp = {
                'channels': 64,
                'num_layers': 5,
                'pe_dim': 8,
                'node_dim': 1,
                'num_heads': 8,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 8,
                'attn_type': 'multihead',
                'forecast_horizon': 36,
                'sequence_length': 12
            }
        hp['num_nodes'] = data.m
        model = GraphTransformer(hp).to(device)
        model_data = {}
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=1e-5)
    
    print("Starting continued training...")
    final_loss = train_operational_model(data, model, optimizer, criterion, hp['batch_size'], device, args.epochs)
    
    print(f"Final training loss after continued training: {final_loss:.6f}")
    
    print("Saving updated operational model...")
    operational_model_data = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hp,
        'training_loss': final_loss,
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'num_time_steps': data.n,
            'num_nodes': data.m,
            'forecast_horizon': data.out_len
        },
        'previous_training_info': model_data.get('timestamp', 'N/A'),
        'previous_training_loss': model_data.get('training_loss', 'N/A')
    }
    
    torch.save(operational_model_data, 'Dissertation/Graph Transformer/model/GraphTransformer/o_model.pt')
    
    print("\n OPERATIONAL MODEL CONTINUED TRAINING COMPLETE ")
    print(f"Updated operational model saved as: model/GraphTransformer/o_model.pt")

if __name__ == "__main__":
    main()