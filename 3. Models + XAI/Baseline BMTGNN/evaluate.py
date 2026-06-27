import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import json
import time
import warnings
from datetime import datetime
import os
import sys
from collections import defaultdict, Counter
import math
import random

warnings.filterwarnings('ignore')

# B-MTGNN ARCHITECTURE 

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        node_dim = x.shape[2]
        if A.shape[-1] != node_dim:
            if A.shape[-1] > node_dim:
                A = A[..., :node_dim]
            else:
                pad = node_dim - A.shape[-1]
                A = F.pad(A, (0, pad))
        if A.shape[0] == node_dim:
            x = torch.einsum('ncwl,ww->ncwl', (x, A))
        else:
            x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self, x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)
        d = d + 1e-5
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        import numbers
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, 
                 static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, 
                 dilation_exponential=1, conv_channels=32, residual_channels=32, 
                 skip_channels=64, end_channels=128, seq_length=12, in_dim=1, 
                 out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.device = device
        self.seq_length = seq_length
        self.node_dim = node_dim
        self.dropout = dropout
        
        self._init_adjacency()
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, 
                                   alpha=tanhalpha, static_feat=static_feat)
        
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1), 
                                              elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1), 
                                              elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, 
                                  kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, 
                                  kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, 
                                  kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, 
                                  kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        
    def _init_adjacency(self):
        if self.gcn_true:
            if self.buildA_true:
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, self.node_dim).to(self.device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(self.node_dim, self.num_nodes).to(self.device), requires_grad=True)
                adp = torch.mm(self.nodevec1, self.nodevec2)
                self.adp = nn.Parameter(F.relu(adp), requires_grad=True)
            else:
                self.adp = nn.Parameter(torch.eye(self.num_nodes).to(self.device), requires_grad=False)
        else:
            self.register_buffer('adp', None)

    def _resize_adjacency(self, adj_tensor, new_size):
        old_size = adj_tensor.size(0)
        requires_grad = adj_tensor.requires_grad
        if new_size > old_size:
            padded = torch.zeros(new_size, new_size, device=adj_tensor.device)
            padded[:old_size, :old_size] = adj_tensor
            return nn.Parameter(padded, requires_grad=requires_grad)
        else:
            return nn.Parameter(adj_tensor[:new_size, :new_size], requires_grad=requires_grad)

    def forward(self, x, idx=None):
        batch_size, input_dim, num_nodes, seq_len = x.shape
    
        if self.gcn_true:
            if not hasattr(self, 'adp') or self.adp is None:
                self.adp = nn.Parameter(torch.eye(num_nodes).to(x.device), requires_grad=False)
            if self.adp.size(0) != num_nodes:
                adj_data = self.adp.data
                resized_adj = self._resize_adjacency(adj_data, num_nodes)
                self.adp = resized_adj
        
        assert seq_len == self.seq_length, f'input sequence length {seq_len} not equal to preset sequence length {self.seq_length}'

        if self.seq_length < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A if self.predefined_A is not None else self.adp
        else:
            adp = None

        x = self.start_conv(x)
        
        if self.seq_length > self.receptive_field:
            skip = self.skip0(x)
        else:
            skip = self.skip0(x[:, :, :, -self.receptive_field:])

        for i in range(self.layers):
            residual = x
            
            filter_out = self.filter_convs[i](x)
            filter_out = torch.tanh(filter_out)
            gate_out = self.gate_convs[i](x)
            gate_out = torch.sigmoid(gate_out)
            x = filter_out * gate_out
            x = F.dropout(x, self.dropout, training=self.training)
            
            s = x
            s = self.skip_convs[i](s)
            skip = skip + s
            
            if self.gcn_true and adp is not None:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

# EVALUATION CLASS 

class BMTGNNEvaluator:
    def __init__(self, model_path, data_file, config=None):
        self.model_path = model_path
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract the actual configuration from the model checkpoint
        self.config = config or {}
        
        # Load model first to extract configuration
        self.extract_config_from_checkpoint()
        
        # Metrics storage
        self.metrics = {}
        
    def extract_config_from_checkpoint(self):
        """Extract model configuration from the checkpoint file"""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the entire checkpoint is the state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Extract configuration from state dict shapes
            self.config = self.infer_config_from_state_dict(state_dict)
            
            print("Inferred configuration from model checkpoint:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            
            # Create model with inferred configuration
            self.model = self.create_model_with_config()
            
            # Load weights with flexible matching
            self.load_weights_with_flexible_matching(state_dict)
            
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def infer_config_from_state_dict(self, state_dict):
        """Infer model configuration from state dict shapes"""
        config = {
            'gcn_true': True,
            'buildA_true': True,
            'num_nodes': 645,
            'seq_length': 12,
            'forecast_horizon': 12,  # Will be updated
            'dropout': 0.3,
            'subgraph_size': 20,
            'node_dim': 40,
            'dilation_exponential': 1,
            'conv_channels': 32,
            'residual_channels': 32,
            'skip_channels': 64,
            'end_channels': 128,
            'layers': 3,
            'propalpha': 0.05,
            'tanhalpha': 3,
            'gcn_depth': 2,
            'normalize': 2,
            'batch_size': 4
        }
        
        # Extract configuration from layer shapes
        for key, tensor in state_dict.items():
            # Infer node_dim from nodevec1 or gc.emb1
            if 'nodevec1' in key and tensor.dim() == 2:
                if tensor.shape[0] == config['num_nodes']:
                    config['node_dim'] = tensor.shape[1]
                    print(f"Inferred node_dim: {config['node_dim']} from {key}")
            
            # Infer forecast_horizon from end_conv_2
            if 'end_conv_2.weight' in key and tensor.dim() == 4:
                config['forecast_horizon'] = tensor.shape[0]
                print(f"Inferred forecast_horizon: {config['forecast_horizon']} from {key}")
            
            # Infer residual_channels from start_conv
            if 'start_conv.weight' in key and tensor.dim() == 4:
                config['residual_channels'] = tensor.shape[0]
                print(f"Inferred residual_channels: {config['residual_channels']} from {key}")
            
            # Infer conv_channels from filter_convs
            if 'filter_convs.0.tconv.0.weight' in key and tensor.dim() == 4:
                # Each tconv outputs cout/4 channels, so multiply by 4
                config['conv_channels'] = tensor.shape[0] * 4
                print(f"Inferred conv_channels: {config['conv_channels']} from {key}")
            
            # Infer skip_channels from skip_convs
            if 'skip_convs.0.weight' in key and tensor.dim() == 4:
                config['skip_channels'] = tensor.shape[0]
                print(f"Inferred skip_channels: {config['skip_channels']} from {key}")
            
            # Infer end_channels from end_conv_1
            if 'end_conv_1.weight' in key and tensor.dim() == 4:
                config['end_channels'] = tensor.shape[0]
                print(f"Inferred end_channels: {config['end_channels']} from {key}")
            
            # Infer layers by counting filter_convs
            if 'filter_convs.' in key and '.tconv.0.weight' in key:
                try:
                    layer_idx = int(key.split('.')[1])
                    config['layers'] = max(config['layers'], layer_idx + 1)
                except:
                    pass
        
        # Infer gcn_depth from gconv1.0.mlp.mlp.weight shape
        for key, tensor in state_dict.items():
            if 'gconv1.0.mlp.mlp.weight' in key and tensor.dim() == 4:
                # The weight shape is [out_channels, (gdep+1)*in_channels, 1, 1]
                # We know out_channels = residual_channels
                # So (gdep+1)*in_channels = tensor.shape[1]
                # And in_channels = conv_channels
                if 'conv_channels' in config:
                    gdep_plus_one = tensor.shape[1] // config['conv_channels']
                    config['gcn_depth'] = gdep_plus_one - 1
                    print(f"Inferred gcn_depth: {config['gcn_depth']} from {key}")
                break
        
        return config
    
    def create_model_with_config(self):
        """Create model with the extracted configuration"""
        model = gtnet(
            gcn_true=self.config['gcn_true'],
            buildA_true=self.config['buildA_true'],
            gcn_depth=self.config['gcn_depth'],
            num_nodes=self.config['num_nodes'],
            device=self.device,
            predefined_A=None,
            dropout=self.config['dropout'],
            subgraph_size=self.config['subgraph_size'],
            node_dim=self.config['node_dim'],
            dilation_exponential=self.config['dilation_exponential'],
            conv_channels=self.config['conv_channels'],
            residual_channels=self.config['residual_channels'],
            skip_channels=self.config['skip_channels'],
            end_channels=self.config['end_channels'],
            seq_length=self.config['seq_length'],
            in_dim=1,
            out_dim=self.config['forecast_horizon'],
            layers=self.config['layers'],
            propalpha=self.config['propalpha'],
            tanhalpha=self.config['tanhalpha']
        ).to(self.device)
        
        return model
    
    def load_weights_with_flexible_matching(self, state_dict):
        """Load weights with flexible matching to handle architecture differences"""
        model_dict = self.model.state_dict()
        
        # Filter out unnecessary keys and handle shape mismatches
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            k_clean = k.replace('module.', '')
            
            if k_clean in model_dict:
                if model_dict[k_clean].shape == v.shape:
                    filtered_state_dict[k_clean] = v
                else:
                    print(f"Shape mismatch for {k_clean}: model {model_dict[k_clean].shape} vs checkpoint {v.shape}")
                    # Try to handle common shape mismatches
                    if len(v.shape) == len(model_dict[k_clean].shape):
                        # For weights, try to truncate or pad
                        if v.numel() >= model_dict[k_clean].numel():
                            # Truncate if checkpoint has more parameters
                            try:
                                reshaped = v.view(model_dict[k_clean].shape)
                                filtered_state_dict[k_clean] = reshaped
                                print(f"  Reshaped {k_clean} from {v.shape} to {model_dict[k_clean].shape}")
                            except:
                                print(f"  Could not reshape {k_clean}")
                        else:
                            # Pad if checkpoint has fewer parameters
                            try:
                                padded = torch.zeros_like(model_dict[k_clean])
                                # Try to copy as much as possible
                                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(padded.shape, v.shape))
                                padded[slices] = v[slices]
                                filtered_state_dict[k_clean] = padded
                                print(f"  Padded {k_clean} from {v.shape} to {model_dict[k_clean].shape}")
                            except:
                                print(f"  Could not pad {k_clean}")
            else:
                # Try alternative naming
                possible_keys = [
                    k_clean,
                    k_clean.replace('gconv1.', 'gconv1.0.'),
                    k_clean.replace('gconv2.', 'gconv2.0.'),
                    k_clean.replace('filter_convs.', 'filter_convs.0.'),
                    k_clean.replace('gate_convs.', 'gate_convs.0.'),
                    k_clean.replace('residual_convs.', 'residual_convs.0.'),
                    k_clean.replace('skip_convs.', 'skip_convs.0.'),
                    k_clean.replace('norm.', 'norm.0.')
                ]
                
                for possible_key in possible_keys:
                    if possible_key in model_dict and model_dict[possible_key].shape == v.shape:
                        filtered_state_dict[possible_key] = v
                        print(f"Matched {k} -> {possible_key}")
                        break
        
        # Update model dict with filtered state dict
        model_dict.update(filtered_state_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(filtered_state_dict)} parameters")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    def load_and_split_data(self):
        """Load data from CSV and split into training and testing periods"""
        try:
            # Load the complete dataset
            print(f"Loading data from {self.data_file}...")
            self.data_df = pd.read_csv(self.data_file)
            
            # Check data structure
            print(f"Data columns: {self.data_df.columns.tolist()}")
            print(f"Data shape: {self.data_df.shape}")
            
            # Extract date information
            date_column = None
            for col in self.data_df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                    date_column = col
                    break
            
            if date_column is None:
                # Assume first column is date
                date_column = self.data_df.columns[0]
                print(f"No explicit date column found, using first column: {date_column}")
            
            # Convert to datetime
            self.data_df['date'] = pd.to_datetime(self.data_df[date_column])
            
            # Extract the actual data (excluding date column)
            data_columns = [col for col in self.data_df.columns if col != date_column and col != 'date']
            self.threat_data = self.data_df[data_columns].values.astype(np.float32)
            
            print(f"Processed data shape: {self.threat_data.shape}")
            print(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
            
            # Use 70% for training, 30% for testing (matching B-MTGNN)
            split_idx = int(0.7 * len(self.threat_data))
            self.data_train = self.threat_data[:split_idx]
            self.data_test = self.threat_data[split_idx:]
            self.dates_train = self.data_df['date'][:split_idx]
            self.dates_test = self.data_df['date'][split_idx:]
            
            print(f"Training data: {self.data_train.shape} (first {split_idx} samples)")
            print(f"Testing data: {self.data_test.shape} (last {len(self.data_test)} samples)")
            
            # Ensure data has correct number of nodes
            if self.data_train.shape[1] != self.config['num_nodes']:
                print(f"Adjusting data dimensions from {self.data_train.shape[1]} to {self.config['num_nodes']} nodes")
                if self.data_train.shape[1] > self.config['num_nodes']:
                    # Truncate if data has more nodes
                    self.data_train = self.data_train[:, :self.config['num_nodes']]
                    self.data_test = self.data_test[:, :self.config['num_nodes']]
                else:
                    # Pad if data has fewer nodes
                    pad_width = self.config['num_nodes'] - self.data_train.shape[1]
                    self.data_train = np.pad(self.data_train, ((0, 0), (0, pad_width)), mode='constant')
                    self.data_test = np.pad(self.data_test, ((0, 0), (0, pad_width)), mode='constant')
            
            # Normalize data using training statistics
            self.data_mean = np.mean(self.data_train, axis=0)
            self.data_std = np.std(self.data_train, axis=0) + 1e-8
            
            self.data_train_norm = (self.data_train - self.data_mean) / self.data_std
            self.data_test_norm = (self.data_test - self.data_mean) / self.data_std
            
            print("Data loaded and normalized successfully")
            
        except Exception as e:
            print(f"Error loading and splitting data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_sequences(self, data, sequence_length, forecast_horizon):
        """Create sequences for B-MTGNN model input"""
        sequences = []
        targets = []
        
        total_possible = len(data) - sequence_length - forecast_horizon + 1
        
        if total_possible <= 0:
            print(f"Warning: Not enough data to generate sequences. Data length: {len(data)}, sequence_length: {sequence_length}")
            return [], []
        
        for i in range(total_possible):
            input_seq = data[i:i+sequence_length]
            target_seq = data[i+sequence_length:i+sequence_length+forecast_horizon]
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        print(f"Generated {len(sequences)} sequences with forecast horizon {forecast_horizon}")
        return sequences, targets
    
    def generate_predictions(self, data):
        """Generate predictions using the B-MTGNN model"""
        sequence_length = self.config['seq_length']
        forecast_horizon = self.config['forecast_horizon']
        
        # Generate sequences
        sequences, actuals = self.create_sequences(data, sequence_length, forecast_horizon)
        
        if len(sequences) == 0:
            print("No sequences generated, returning empty predictions")
            return np.array([]), np.array([])
        
        predictions = []
        actual_values = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(sequences), self.config.get('batch_size', 4)):
                batch_sequences = sequences[i:i+self.config.get('batch_size', 4)]
                batch_actuals = actuals[i:i+self.config.get('batch_size', 4)]
                
                if len(batch_sequences) == 0:
                    continue
                
                # Prepare input tensor
                X = torch.FloatTensor(np.array(batch_sequences)).to(self.device)
                X = X.unsqueeze(1)  # Add channel dimension
                X = X.transpose(2, 3)  # B-MTGNN expects [batch, channels, nodes, seq_len]
                
                # Generate predictions
                pred = self.model(X)
                
                # Handle output shape
                if len(pred.shape) == 4:
                    pred = pred.squeeze(3)  # Remove last dimension if needed
                
                # Move to CPU and convert to numpy
                pred_np = pred.cpu().numpy()
                actual_np = np.array(batch_actuals)
                
                # Ensure shapes match
                if pred_np.shape[1] == forecast_horizon and pred_np.shape[2] == self.config['num_nodes']:
                    pred_np = np.transpose(pred_np, (0, 2, 1))
                
                predictions.append(pred_np)
                actual_values.append(actual_np)
        
        if len(predictions) == 0:
            return np.array([]), np.array([])
        
        # Concatenate all predictions
        predictions_array = np.concatenate(predictions, axis=0)
        actuals_array = np.concatenate(actual_values, axis=0)
        
        print(f"Generated predictions shape: {predictions_array.shape}")
        print(f"Actuals shape: {actuals_array.shape}")
        
        return predictions_array, actuals_array
    
    def calculate_msmape(self, predictions, actuals):
        """Calculate M-SMAPE (Modified Symmetric Mean Absolute Percentage Error)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        # Align shapes for broadcasting
        predictions = self._align_shape(predictions, actuals)
        epsilon = 1e-8
        numerator = np.abs(predictions - actuals)
        denominator = (np.abs(predictions) + np.abs(actuals)) / 2 + epsilon
        smape = 2 * np.mean(numerator / denominator)
        return smape * 100
    
    def calculate_mae(self, predictions, actuals):
        """Calculate Mean Absolute Error (MAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        predictions = self._align_shape(predictions, actuals)
        mae = np.mean(np.abs(predictions - actuals))
        return mae
    
    def calculate_mape(self, predictions, actuals):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        predictions = self._align_shape(predictions, actuals)
        epsilon = 1e-8
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        return mape
    
    def calculate_rmse(self, predictions, actuals):
        """Calculate Root Mean Squared Error (RMSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        predictions = self._align_shape(predictions, actuals)
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        return rmse
    
    def calculate_rae(self, predictions, actuals):
        """Calculate Relative Absolute Error (RAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        predictions = self._align_shape(predictions, actuals)
        sum_absolute_errors = np.sum(np.abs(predictions - actuals))
        mean_actuals = np.mean(actuals)
        sum_absolute_deviation = np.sum(np.abs(actuals - mean_actuals))
        if sum_absolute_deviation == 0:
            return float('inf')
        rae = sum_absolute_errors / sum_absolute_deviation
        return rae
    
    def calculate_rrse(self, predictions, actuals):
        """Calculate Root Relative Squared Error (RRSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
        predictions = self._align_shape(predictions, actuals)
        sum_squared_errors = np.sum((predictions - actuals) ** 2)
        mean_actuals = np.mean(actuals)
        sum_squared_deviation = np.sum((actuals - mean_actuals) ** 2)
        if sum_squared_deviation == 0:
            return float('inf')
        rrse = np.sqrt(sum_squared_errors / sum_squared_deviation)
        return rrse

    def _align_shape(self, arr, ref):
        """
        Align arr to the shape of ref for elementwise operations.
        If arr and ref have the same set of dimensions but in different orders, transpose arr.
        """
        if arr.shape == ref.shape:
            return arr
        # Try to find a permutation that matches ref's shape
        if sorted(arr.shape) == sorted(ref.shape):
            # Find permutation
            for axes in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                if tuple(arr.shape[ax] for ax in axes) == ref.shape:
                    return np.transpose(arr, axes)
        # If 3D and last two axes swapped
        if arr.ndim == 3 and ref.ndim == 3 and arr.shape[0] == ref.shape[0] and arr.shape[2] == ref.shape[1] and arr.shape[1] == ref.shape[2]:
            return np.transpose(arr, (0,2,1))
        # If 2D and axes swapped
        if arr.ndim == 2 and ref.ndim == 2 and arr.shape[0] == ref.shape[1] and arr.shape[1] == ref.shape[0]:
            return arr.T
        # If cannot align, return arr and let numpy raise error
        return arr

    #  EVALUATION CLASS 

    class BMTGNNEvaluator:
        def __init__(self, model_path, data_file, config=None):
            self.model_path = model_path
            self.data_file = data_file
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Extract the actual configuration from the model checkpoint
            self.config = config or {}
            
            # Load model first to extract configuration
            self.extract_config_from_checkpoint()
            
            # Metrics storage
            self.metrics = {}
            
        def extract_config_from_checkpoint(self):
            """Extract model configuration from the checkpoint file"""
            try:
                print(f"Loading model from {self.model_path}...")
                
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # Assume the entire checkpoint is the state dict
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Extract configuration from state dict shapes
                self.config = self.infer_config_from_state_dict(state_dict)
                
                print("Inferred configuration from model checkpoint:")
                for key, value in self.config.items():
                    print(f"  {key}: {value}")
                
                # Create model with inferred configuration
                self.model = self.create_model_with_config()
                
                # Load weights with flexible matching
                self.load_weights_with_flexible_matching(state_dict)
                
                self.model.eval()
                print("Model loaded successfully")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        def infer_config_from_state_dict(self, state_dict):
            """Infer model configuration from state dict shapes"""
            config = {
                'gcn_true': True,
                'buildA_true': True,
                'num_nodes': 645,
                'seq_length': 12,
                'forecast_horizon': 12,  # Will be updated
                'dropout': 0.3,
                'subgraph_size': 20,
                'node_dim': 40,
                'dilation_exponential': 1,
                'conv_channels': 32,
                'residual_channels': 32,
                'skip_channels': 64,
                'end_channels': 128,
                'layers': 3,
                'propalpha': 0.05,
                'tanhalpha': 3,
                'gcn_depth': 2,
                'normalize': 2,
                'batch_size': 4
            }
            
            # Extract configuration from layer shapes
            for key, tensor in state_dict.items():
                # Infer node_dim from nodevec1 or gc.emb1
                if 'nodevec1' in key and tensor.dim() == 2:
                    if tensor.shape[0] == config['num_nodes']:
                        config['node_dim'] = tensor.shape[1]
                        print(f"Inferred node_dim: {config['node_dim']} from {key}")
                
                # Infer forecast_horizon from end_conv_2
                if 'end_conv_2.weight' in key and tensor.dim() == 4:
                    config['forecast_horizon'] = tensor.shape[0]
                    print(f"Inferred forecast_horizon: {config['forecast_horizon']} from {key}")
                
                # Infer residual_channels from start_conv
                if 'start_conv.weight' in key and tensor.dim() == 4:
                    config['residual_channels'] = tensor.shape[0]
                    print(f"Inferred residual_channels: {config['residual_channels']} from {key}")
                
                # Infer conv_channels from filter_convs
                if 'filter_convs.0.tconv.0.weight' in key and tensor.dim() == 4:
                    # Each tconv outputs cout/4 channels, so multiply by 4
                    config['conv_channels'] = tensor.shape[0] * 4
                    print(f"Inferred conv_channels: {config['conv_channels']} from {key}")
                
                # Infer skip_channels from skip_convs
                if 'skip_convs.0.weight' in key and tensor.dim() == 4:
                    config['skip_channels'] = tensor.shape[0]
                    print(f"Inferred skip_channels: {config['skip_channels']} from {key}")
                
                # Infer end_channels from end_conv_1
                if 'end_conv_1.weight' in key and tensor.dim() == 4:
                    config['end_channels'] = tensor.shape[0]
                    print(f"Inferred end_channels: {config['end_channels']} from {key}")
                
                # Infer layers by counting filter_convs
                if 'filter_convs.' in key and '.tconv.0.weight' in key:
                    try:
                        layer_idx = int(key.split('.')[1])
                        config['layers'] = max(config['layers'], layer_idx + 1)
                    except:
                        pass
            
            # Infer gcn_depth from gconv1.0.mlp.mlp.weight shape
            for key, tensor in state_dict.items():
                if 'gconv1.0.mlp.mlp.weight' in key and tensor.dim() == 4:
                    # The weight shape is [out_channels, (gdep+1)*in_channels, 1, 1]
                    # We know out_channels = residual_channels
                    # So (gdep+1)*in_channels = tensor.shape[1]
                    # And in_channels = conv_channels
                    if 'conv_channels' in config:
                        gdep_plus_one = tensor.shape[1] // config['conv_channels']
                        config['gcn_depth'] = gdep_plus_one - 1
                        print(f"Inferred gcn_depth: {config['gcn_depth']} from {key}")
                    break
            
            return config
        
        def create_model_with_config(self):
            """Create model with the extracted configuration"""
            model = gtnet(
                gcn_true=self.config['gcn_true'],
                buildA_true=self.config['buildA_true'],
                gcn_depth=self.config['gcn_depth'],
                num_nodes=self.config['num_nodes'],
                device=self.device,
                predefined_A=None,
                dropout=self.config['dropout'],
                subgraph_size=self.config['subgraph_size'],
                node_dim=self.config['node_dim'],
                dilation_exponential=self.config['dilation_exponential'],
                conv_channels=self.config['conv_channels'],
                residual_channels=self.config['residual_channels'],
                skip_channels=self.config['skip_channels'],
                end_channels=self.config['end_channels'],
                seq_length=self.config['seq_length'],
                in_dim=1,
                out_dim=self.config['forecast_horizon'],
                layers=self.config['layers'],
                propalpha=self.config['propalpha'],
                tanhalpha=self.config['tanhalpha']
            ).to(self.device)
            
            return model
        
        def load_weights_with_flexible_matching(self, state_dict):
            """Load weights with flexible matching to handle architecture differences"""
            model_dict = self.model.state_dict()
            
            # Filter out unnecessary keys and handle shape mismatches
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present (from DataParallel)
                k_clean = k.replace('module.', '')
                
                if k_clean in model_dict:
                    if model_dict[k_clean].shape == v.shape:
                        filtered_state_dict[k_clean] = v
                    else:
                        print(f"Shape mismatch for {k_clean}: model {model_dict[k_clean].shape} vs checkpoint {v.shape}")
                        # Try to handle common shape mismatches
                        if len(v.shape) == len(model_dict[k_clean].shape):
                            # For weights, try to truncate or pad
                            if v.numel() >= model_dict[k_clean].numel():
                                # Truncate if checkpoint has more parameters
                                try:
                                    reshaped = v.view(model_dict[k_clean].shape)
                                    filtered_state_dict[k_clean] = reshaped
                                    print(f"  Reshaped {k_clean} from {v.shape} to {model_dict[k_clean].shape}")
                                except:
                                    print(f"  Could not reshape {k_clean}")
                            else:
                                # Pad if checkpoint has fewer parameters
                                try:
                                    padded = torch.zeros_like(model_dict[k_clean])
                                    # Try to copy as much as possible
                                    slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(padded.shape, v.shape))
                                    padded[slices] = v[slices]
                                    filtered_state_dict[k_clean] = padded
                                    print(f"  Padded {k_clean} from {v.shape} to {model_dict[k_clean].shape}")
                                except:
                                    print(f"  Could not pad {k_clean}")
                else:
                    # Try alternative naming
                    possible_keys = [
                        k_clean,
                        k_clean.replace('gconv1.', 'gconv1.0.'),
                        k_clean.replace('gconv2.', 'gconv2.0.'),
                        k_clean.replace('filter_convs.', 'filter_convs.0.'),
                        k_clean.replace('gate_convs.', 'gate_convs.0.'),
                        k_clean.replace('residual_convs.', 'residual_convs.0.'),
                        k_clean.replace('skip_convs.', 'skip_convs.0.'),
                        k_clean.replace('norm.', 'norm.0.')
                    ]
                    
                    for possible_key in possible_keys:
                        if possible_key in model_dict and model_dict[possible_key].shape == v.shape:
                            filtered_state_dict[possible_key] = v
                            print(f"Matched {k} -> {possible_key}")
                            break
            
            # Update model dict with filtered state dict
            model_dict.update(filtered_state_dict)
            missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(filtered_state_dict)} parameters")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        def load_and_split_data(self):
            """Load data from CSV and split into training and testing periods"""
            try:
                # Load the complete dataset
                print(f"Loading data from {self.data_file}...")
                self.data_df = pd.read_csv(self.data_file)
                
                # Check data structure
                print(f"Data columns: {self.data_df.columns.tolist()}")
                print(f"Data shape: {self.data_df.shape}")
                
                # Extract date information
                date_column = None
                for col in self.data_df.columns:
                    if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                        date_column = col
                        break
                
                if date_column is None:
                    # Assume first column is date
                    date_column = self.data_df.columns[0]
                    print(f"No explicit date column found, using first column: {date_column}")
                
                # Convert to datetime
                self.data_df['date'] = pd.to_datetime(self.data_df[date_column])
                
                # Extract the actual data (excluding date column)
                data_columns = [col for col in self.data_df.columns if col != date_column and col != 'date']
                self.threat_data = self.data_df[data_columns].values.astype(np.float32)
                
                print(f"Processed data shape: {self.threat_data.shape}")
                print(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
                
                # Use 70% for training, 30% for testing (matching B-MTGNN)
                split_idx = int(0.7 * len(self.threat_data))
                self.data_train = self.threat_data[:split_idx]
                self.data_test = self.threat_data[split_idx:]
                self.dates_train = self.data_df['date'][:split_idx]
                self.dates_test = self.data_df['date'][split_idx:]
                
                print(f"Training data: {self.data_train.shape} (first {split_idx} samples)")
                print(f"Testing data: {self.data_test.shape} (last {len(self.data_test)} samples)")
                
                # Ensure data has correct number of nodes
                if self.data_train.shape[1] != self.config['num_nodes']:
                    print(f"Adjusting data dimensions from {self.data_train.shape[1]} to {self.config['num_nodes']} nodes")
                    if self.data_train.shape[1] > self.config['num_nodes']:
                        # Truncate if data has more nodes
                        self.data_train = self.data_train[:, :self.config['num_nodes']]
                        self.data_test = self.data_test[:, :self.config['num_nodes']]
                    else:
                        # Pad if data has fewer nodes
                        pad_width = self.config['num_nodes'] - self.data_train.shape[1]
                        self.data_train = np.pad(self.data_train, ((0, 0), (0, pad_width)), mode='constant')
                        self.data_test = np.pad(self.data_test, ((0, 0), (0, pad_width)), mode='constant')
                
                # Normalize data using training statistics
                self.data_mean = np.mean(self.data_train, axis=0)
                self.data_std = np.std(self.data_train, axis=0) + 1e-8
                
                self.data_train_norm = (self.data_train - self.data_mean) / self.data_std
                self.data_test_norm = (self.data_test - self.data_mean) / self.data_std
                
                print("Data loaded and normalized successfully")
                
            except Exception as e:
                print(f"Error loading and splitting data: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        def create_sequences(self, data, sequence_length, forecast_horizon):
            """Create sequences for B-MTGNN model input"""
            sequences = []
            targets = []
            
            total_possible = len(data) - sequence_length - forecast_horizon + 1
            
            if total_possible <= 0:
                print(f"Warning: Not enough data to generate sequences. Data length: {len(data)}, sequence_length: {sequence_length}")
                return [], []
            
            for i in range(total_possible):
                input_seq = data[i:i+sequence_length]
                target_seq = data[i+sequence_length:i+sequence_length+forecast_horizon]
                
                sequences.append(input_seq)
                targets.append(target_seq)
            
            print(f"Generated {len(sequences)} sequences with forecast horizon {forecast_horizon}")
            return sequences, targets
        
        def generate_predictions(self, data):
            """Generate predictions using the B-MTGNN model"""
            sequence_length = self.config['seq_length']
            forecast_horizon = self.config['forecast_horizon']
            
            # Generate sequences
            sequences, actuals = self.create_sequences(data, sequence_length, forecast_horizon)
            
            if len(sequences) == 0:
                print("No sequences generated, returning empty predictions")
                return np.array([]), np.array([])
            
            predictions = []
            actual_values = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(sequences), self.config.get('batch_size', 4)):
                    batch_sequences = sequences[i:i+self.config.get('batch_size', 4)]
                    batch_actuals = actuals[i:i+self.config.get('batch_size', 4)]
                    
                    if len(batch_sequences) == 0:
                        continue
                    
                    # Prepare input tensor
                    X = torch.FloatTensor(np.array(batch_sequences)).to(self.device)
                    X = X.unsqueeze(1)  # Add channel dimension
                    X = X.transpose(2, 3)  # B-MTGNN expects [batch, channels, nodes, seq_len]
                    
                    # Generate predictions
                    pred = self.model(X)
                    
                    # Handle output shape
                    if len(pred.shape) == 4:
                        pred = pred.squeeze(3)  # Remove last dimension if needed
                    
                    # Move to CPU and convert to numpy
                    pred_np = pred.cpu().numpy()
                    actual_np = np.array(batch_actuals)
                    
                    # Ensure shapes match
                    if pred_np.shape[1] == forecast_horizon and pred_np.shape[2] == self.config['num_nodes']:
                        pred_np = np.transpose(pred_np, (0, 2, 1))
                    
                    predictions.append(pred_np)
                    actual_values.append(actual_np)
            
            if len(predictions) == 0:
                return np.array([]), np.array([])
            
            # Concatenate all predictions
            predictions_array = np.concatenate(predictions, axis=0)
            actuals_array = np.concatenate(actual_values, axis=0)
            
            print(f"Generated predictions shape: {predictions_array.shape}")
            print(f"Actuals shape: {actuals_array.shape}")
            
            return predictions_array, actuals_array
        
        def calculate_msmape(self, predictions, actuals):
            """Calculate M-SMAPE (Modified Symmetric Mean Absolute Percentage Error)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            # Align shapes for broadcasting
            predictions = self._align_shape(predictions, actuals)
            epsilon = 1e-8
            numerator = np.abs(predictions - actuals)
            denominator = (np.abs(predictions) + np.abs(actuals)) / 2 + epsilon
            smape = 2 * np.mean(numerator / denominator)
            return smape * 100
        
        def calculate_mae(self, predictions, actuals):
            """Calculate Mean Absolute Error (MAE)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            predictions = self._align_shape(predictions, actuals)
            mae = np.mean(np.abs(predictions - actuals))
            return mae
        
        def calculate_mape(self, predictions, actuals):
            """Calculate Mean Absolute Percentage Error (MAPE)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            predictions = self._align_shape(predictions, actuals)
            epsilon = 1e-8
            mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
            return mape
        
        def calculate_rmse(self, predictions, actuals):
            """Calculate Root Mean Squared Error (RMSE)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            predictions = self._align_shape(predictions, actuals)
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            return rmse
        
        def calculate_rae(self, predictions, actuals):
            """Calculate Relative Absolute Error (RAE)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            predictions = self._align_shape(predictions, actuals)
            sum_absolute_errors = np.sum(np.abs(predictions - actuals))
            mean_actuals = np.mean(actuals)
            sum_absolute_deviation = np.sum(np.abs(actuals - mean_actuals))
            if sum_absolute_deviation == 0:
                return float('inf')
            rae = sum_absolute_errors / sum_absolute_deviation
            return rae
        
        def calculate_rrse(self, predictions, actuals):
            """Calculate Root Relative Squared Error (RRSE)"""
            if predictions.size == 0 or actuals.size == 0:
                return float('inf')
            predictions = self._align_shape(predictions, actuals)
            sum_squared_errors = np.sum((predictions - actuals) ** 2)
            mean_actuals = np.mean(actuals)
            sum_squared_deviation = np.sum((actuals - mean_actuals) ** 2)
            if sum_squared_deviation == 0:
                return float('inf')
            rrse = np.sqrt(sum_squared_errors / sum_squared_deviation)
            return rrse
        
        def _align_shape(self, arr, ref):
            """
            Align arr to the shape of ref for elementwise operations.
            If arr and ref have the same set of dimensions but in different orders, transpose arr.
            """
            if arr.shape == ref.shape:
                return arr
            # Try to find a permutation that matches ref's shape
            if sorted(arr.shape) == sorted(ref.shape):
                # Find permutation
                for axes in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                    if tuple(arr.shape[ax] for ax in axes) == ref.shape:
                        return np.transpose(arr, axes)
            # If 3D and last two axes swapped
            if arr.ndim == 3 and ref.ndim == 3 and arr.shape[0] == ref.shape[0] and arr.shape[2] == ref.shape[1] and arr.shape[1] == ref.shape[2]:
                return np.transpose(arr, (0,2,1))
            # If 2D and axes swapped
            if arr.ndim == 2 and ref.ndim == 2 and arr.shape[0] == ref.shape[1] and arr.shape[1] == ref.shape[0]:
                return arr.T
            # If cannot align, return arr and let numpy raise error
            return arr

    def calculate_auc_roc(self, predictions, actuals, threshold=0.3):
        """Calculate AUC-ROC for binary threat detection"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.5, np.array([0, 1]), np.array([0, 1])
            
        # Convert to binary classification (threat vs no-threat)
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        # Flatten for overall AUC
        pred_flat = pred_binary.flatten()
        actual_flat = actual_binary.flatten()
        
        try:
            auc = roc_auc_score(actual_flat, pred_flat)
            fpr, tpr, _ = roc_curve(actual_flat, pred_flat)
            return auc, fpr, tpr
        except:
            return 0.5, np.array([0, 1]), np.array([0, 1])
    
    def calculate_false_positive_rate(self, predictions, actuals, threshold=0.3):
        """Calculate False Positive Rate"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        fp = np.sum((pred_binary == 1) & (actual_binary == 0))
        tn = np.sum((pred_binary == 0) & (actual_binary == 0))
        
        fpr = fp / (fp + tn + 1e-8)
        return fpr
    
    def calculate_attack_coverage(self, predictions, actuals, threshold=0.3):
        """Calculate Attack Coverage percentage"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        total_attacks = np.sum(actuals > threshold)
        detected_attacks = np.sum((predictions > threshold) & (actuals > threshold))
        
        coverage = detected_attacks / (total_attacks + 1e-8)
        return coverage
    
    def calculate_alert_precision(self, predictions, actuals, threshold=0.3):
        """Calculate Alert Precision"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        true_positives = np.sum((predictions > threshold) & (actuals > threshold))
        false_positives = np.sum((predictions > threshold) & (actuals <= threshold))
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        return precision
    
    def calculate_directional_accuracy(self, predictions, actuals):
        """Calculate Directional Accuracy for trend prediction"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        # For multi-step predictions
        if len(predictions.shape) == 3:
            # Compare trends across time steps
            pred_trend = np.diff(predictions, axis=1) > 0
            actual_trend = np.diff(actuals, axis=1) > 0
            correct_direction = (pred_trend == actual_trend)
        else:
            # For flattened arrays
            pred_trend = np.diff(predictions, axis=0) > 0
            actual_trend = np.diff(actuals, axis=0) > 0
            correct_direction = (pred_trend == actual_trend)
        
        directional_accuracy = np.mean(correct_direction)
        return directional_accuracy
    
    def calculate_quantile_loss(self, predictions, actuals, quantiles=[0.1, 0.5, 0.9]):
        """Calculate Quantile Loss (Pinball Loss)"""
        if predictions.size == 0 or actuals.size == 0:
            return {f'quantile_{int(q*100)}': float('inf') for q in quantiles}
            
        losses = {}
        
        for q in quantiles:
            error = actuals - predictions
            loss = np.maximum(q * error, (q - 1) * error)
            losses[f'quantile_{int(q*100)}'] = np.mean(loss)
        
        return losses
    
    def calculate_ks_test(self, predictions, actuals):
        """Calculate Kolmogorov-Smirnov test statistics"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0, 0.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        stat, p_value = ks_2samp(pred_flat, actual_flat)
        return stat, p_value
    
    def calculate_js_divergence(self, predictions, actuals, bins=50):
        """Calculate Jensen-Shannon Divergence"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        # Create histograms
        min_val = min(np.min(pred_flat), np.min(actual_flat))
        max_val = max(np.max(pred_flat), np.max(actual_flat))
        
        pred_hist, _ = np.histogram(pred_flat, bins=bins, range=(min_val, max_val), density=True)
        actual_hist, _ = np.histogram(actual_flat, bins=bins, range=(min_val, max_val), density=True)
        
        js_div = jensenshannon(pred_hist, actual_hist)
        return js_div
    
    def calculate_temporal_correlation(self, predictions, actuals):
        """Calculate Temporal Correlation Coefficient"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        correlations = []
        
        # Handle multi-step predictions
        if len(predictions.shape) == 3:
            for node_idx in range(actuals.shape[2]):
                node_actual = actuals[:, :, node_idx].flatten()
                node_pred = predictions[:, :, node_idx].flatten()
                
                # Remove constant sequences
                if np.std(node_actual) > 0 and np.std(node_pred) > 0:
                    corr = np.corrcoef(node_actual, node_pred)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        else:
            # Single correlation for flattened arrays
            pred_flat = predictions.flatten()
            actual_flat = actuals.flatten()
            if np.std(actual_flat) > 0 and np.std(pred_flat) > 0:
                corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0
    
    def initialize_placeholder_metrics(self):
        """Initialize metrics with placeholder values when no predictions are generated"""
        placeholder_metrics = {
            'M-SMAPE_Train': float('inf'),
            'M-SMAPE_Test': float('inf'),
            'MAE_Train': float('inf'),
            'MAE_Test': float('inf'),
            'MAPE_Train': float('inf'),
            'MAPE_Test': float('inf'),
            'RMSE_Train': float('inf'),
            'RMSE_Test': float('inf'),
            'RAE_Train': float('inf'),
            'RAE_Test': float('inf'),
            'RRSE_Train': float('inf'),
            'RRSE_Test': float('inf'),
            'AUC_ROC_Train': 0.5,
            'AUC_ROC_Test': 0.5,
            'FPR_Train': 1.0,
            'FPR_Test': 1.0,
            'Attack_Coverage_Train': 0.0,
            'Attack_Coverage_Test': 0.0,
            'Alert_Precision_Train': 0.0,
            'Alert_Precision_Test': 0.0,
            'Directional_Accuracy_Train': 0.0,
            'Directional_Accuracy_Test': 0.0,
            'Quantile_Loss_Train_quantile_10': float('inf'),
            'Quantile_Loss_Train_quantile_50': float('inf'),
            'Quantile_Loss_Train_quantile_90': float('inf'),
            'Quantile_Loss_Test_quantile_10': float('inf'),
            'Quantile_Loss_Test_quantile_50': float('inf'),
            'Quantile_Loss_Test_quantile_90': float('inf'),
            'KS_Statistic_Train': 1.0,
            'KS_Statistic_Test': 1.0,
            'KS_p_value_Train': 0.0,
            'KS_p_value_Test': 0.0,
            'JS_Divergence_Train': 1.0,
            'JS_Divergence_Test': 1.0,
            'Temporal_Correlation_Train': 0.0,
            'Temporal_Correlation_Test': 0.0,
        }
        self.metrics.update(placeholder_metrics)

    def plot_detailed_analysis(self):
        """Create detailed analysis plots including new metrics"""
        if not hasattr(self, 'evaluation_data') or self.evaluation_data['predictions_test'].size == 0:
            print("No evaluation data available for plotting")
            return
        
        # Create a larger figure for more subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()
        
        # 1. ROC Curves
        axes[0].plot(self.evaluation_data['fpr_curve_train'], 
                    self.evaluation_data['tpr_curve_train'], 
                    'b-', linewidth=2, label=f'Train (AUC = {self.metrics["AUC_ROC_Train"]:.3f})')
        axes[0].plot(self.evaluation_data['fpr_curve_test'], 
                    self.evaluation_data['tpr_curve_test'], 
                    'r-', linewidth=2, label=f'Test (AUC = {self.metrics["AUC_ROC_Test"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves: Train vs Test')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Performance Comparison Bar Chart (with new metrics)
        comparison_metrics = ['M-SMAPE_Train', 'M-SMAPE_Test', 
                             'MAE_Train', 'MAE_Test',
                             'MAPE_Train', 'MAPE_Test',
                             'RMSE_Train', 'RMSE_Test',
                             'RAE_Train', 'RAE_Test',
                             'RRSE_Train', 'RRSE_Test']
        
        # Filter out infinite values for plotting
        valid_metrics = []
        valid_values = []
        for m in comparison_metrics:
            if m in self.metrics and self.metrics[m] != float('inf'):
                valid_metrics.append(m)
                valid_values.append(self.metrics[m])
        
        x_pos = np.arange(len(valid_metrics))
        axes[1].bar(x_pos, valid_values, alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(valid_metrics, rotation=45, ha='right', fontsize=9)
        axes[1].set_title('Key Metrics Comparison: Train vs Test')
        axes[1].set_ylabel('Error Value')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Error Metrics Comparison
        error_metrics = ['MAE', 'RMSE', 'RAE', 'RRSE']
        train_errors = [self.metrics.get(f'{m}_Train', 0) for m in error_metrics]
        test_errors = [self.metrics.get(f'{m}_Test', 0) for m in error_metrics]
        
        # Handle infinite values
        train_errors = [0 if np.isinf(x) else x for x in train_errors]
        test_errors = [0 if np.isinf(x) else x for x in test_errors]
        
        x_error = np.arange(len(error_metrics))
        width = 0.35
        axes[2].bar(x_error - width/2, train_errors, width, label='Train', alpha=0.7)
        axes[2].bar(x_error + width/2, test_errors, width, label='Test', alpha=0.7)
        axes[2].set_xticks(x_error)
        axes[2].set_xticklabels(error_metrics)
        axes[2].set_ylabel('Error Value')
        axes[2].set_title('Error Metrics Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Quantile Loss Comparison
        #quantile_metrics_train = [self.metrics['Quantile_Loss_Train_quantile_10'],
         #                       self.metrics['Quantile_Loss_Train_quantile_50'],
          #                      self.metrics['Quantile_Loss_Train_quantile_90']]
        #quantile_metrics_test = [self.metrics['Quantile_Loss_Test_quantile_10'],
         #                       self.metrics['Quantile_Loss_Test_quantile_50'],
          #                      self.metrics['Quantile_Loss_Test_quantile_90']]
        
        # Handle infinite values
        #quantile_metrics_train = [0 if np.isinf(x) else x for x in quantile_metrics_train]
        #quantile_metrics_test = [0 if np.isinf(x) else x for x in quantile_metrics_test]
        
        #x_quantile = np.arange(3)
        #axes[3].bar(x_quantile - width/2, quantile_metrics_train, width, label='Train', alpha=0.7)
        #axes[3].bar(x_quantile + width/2, quantile_metrics_test, width, label='Test', alpha=0.7)
        #axes[3].set_xticks(x_quantile)
        #axes[3].set_xticklabels(['Q10', 'Q50', 'Q90'])
        #axes[3].set_ylabel('Quantile Loss')
        #axes[3].set_title('Quantile Loss Comparison')
        #axes[3].legend()
        # axes[3].grid(True, alpha=0.3)
        
        # 5. Prediction vs Actual Scatter
        if self.evaluation_data['predictions_test'].size > 0:
            test_pred_flat = self.evaluation_data['predictions_test'].flatten()
            test_actual_flat = self.evaluation_data['actuals_test'].flatten()
            
            # Take a sample if too many points
            if len(test_pred_flat) > 10000:
                indices = np.random.choice(len(test_pred_flat), 10000, replace=False)
                test_pred_flat = test_pred_flat[indices]
                test_actual_flat = test_actual_flat[indices]
            
            axes[4].scatter(test_actual_flat, test_pred_flat, alpha=0.1, s=1)
            axes[4].plot([test_actual_flat.min(), test_actual_flat.max()], 
                        [test_actual_flat.min(), test_actual_flat.max()], 'r--', alpha=0.8, linewidth=2)
            axes[4].set_xlabel('Actual Values')
            axes[4].set_ylabel('Predicted Values')
            axes[4].set_title('Prediction vs Actual (Test Set)')
            axes[4].grid(True, alpha=0.3)
        
        # 6. Error Distribution Histogram
        if self.evaluation_data['predictions_test'].size > 0:
            test_errors = test_actual_flat - test_pred_flat
            axes[5].hist(test_errors, bins=50, alpha=0.7, edgecolor='black')
            axes[5].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[5].set_xlabel('Prediction Error (Actual - Predicted)')
            axes[5].set_ylabel('Frequency')
            axes[5].set_title('Error Distribution Histogram (Test Set)')
            axes[5].grid(True, alpha=0.3)
            axes[5].text(0.05, 0.95, f'Mean Error: {np.mean(test_errors):.4f}\nStd Error: {np.std(test_errors):.4f}',
                        transform=axes[5].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('b_mtgnn_baseline_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Comprehensive analysis plot saved as 'b_mtgnn_comprehensive_analysis.png'")
    
    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics including RRSE, RMSE, and RAE"""
        print("Starting comprehensive model evaluation...")
        
        # Load and split data
        self.load_and_split_data()
        
        # Generate predictions for training and testing data
        print(f"\nGenerating predictions for training data...")
        pred_train, actual_train = self.generate_predictions(self.data_train_norm)
        
        print(f"\nGenerating predictions for testing data...")
        pred_test, actual_test = self.generate_predictions(self.data_test_norm)
        
        # Handle empty predictions
        if pred_test.size == 0:
            print("Warning: No predictions generated. Evaluation will use placeholder metrics.")
            self.initialize_placeholder_metrics()
            return self.metrics
        
        # Denormalize predictions and actuals
        #pred_train_denorm = pred_train * self.data_std + self.data_mean
        #actual_train_denorm = actual_train * self.data_std + self.data_mean
        #pred_test_denorm = pred_test * self.data_std + self.data_mean
        #actual_test_denorm = actual_test * self.data_std + self.data_mean
        
        print(f"Training predictions shape: {pred_train.shape}")
        print(f"Testing predictions shape: {pred_test.shape}")
        
        # Calculate metrics
        threshold = 0.3
        
        # 1. Accuracy Metrics
        self.metrics['M-SMAPE_Train'] = self.calculate_msmape(pred_train, actual_train)
        self.metrics['M-SMAPE_Test'] = self.calculate_msmape(pred_test, actual_test)
        
        self.metrics['MAE_Train'] = self.calculate_mae(pred_train, actual_train)
        self.metrics['MAE_Test'] = self.calculate_mae(pred_test, actual_test)
        
        self.metrics['MAPE_Train'] = self.calculate_mape(pred_train, actual_train)
        self.metrics['MAPE_Test'] = self.calculate_mape(pred_test, actual_test)
        
        # NEW: RMSE, RAE, RRSE metrics
        self.metrics['RMSE_Train'] = self.calculate_rmse(pred_train, actual_train)
        self.metrics['RMSE_Test'] = self.calculate_rmse(pred_test, actual_test)
        
        self.metrics['RAE_Train'] = self.calculate_rae(pred_train, actual_train)
        self.metrics['RAE_Test'] = self.calculate_rae(pred_test, actual_test)
        
        self.metrics['RRSE_Train'] = self.calculate_rrse(pred_train, actual_train)
        self.metrics['RRSE_Test'] = self.calculate_rrse(pred_test, actual_test)
        
        # 2. Detection Quality Metrics
        auc_train, fpr_curve_train, tpr_curve_train = self.calculate_auc_roc(pred_train, actual_train, threshold)
        auc_test, fpr_curve_test, tpr_curve_test = self.calculate_auc_roc(pred_test, actual_test, threshold)
        self.metrics['AUC_ROC_Train'] = auc_train
        self.metrics['AUC_ROC_Test'] = auc_test
        
        #self.metrics['FPR_Train'] = self.calculate_false_positive_rate(pred_train, actual_train, threshold)
        #self.metrics['FPR_Test'] = self.calculate_false_positive_rate(pred_test, actual_test, threshold)
        
        # 3. Coverage and Precision
        #self.metrics['Attack_Coverage_Train'] = self.calculate_attack_coverage(pred_train, actual_train, threshold)
        #self.metrics['Attack_Coverage_Test'] = self.calculate_attack_coverage(pred_test, actual_test, threshold)
        
        #self.metrics['Alert_Precision_Train'] = self.calculate_alert_precision(pred_train, actual_train, threshold)
        #self.metrics['Alert_Precision_Test'] = self.calculate_alert_precision(pred_test, actual_test, threshold)
        
        # 4. Distribution Metrics
        #self.metrics['Directional_Accuracy_Train'] = self.calculate_directional_accuracy(pred_train, actual_train)
        #self.metrics['Directional_Accuracy_Test'] = self.calculate_directional_accuracy(pred_test, actual_test)
        
        # Quantile Loss
        #quantile_loss_train = self.calculate_quantile_loss(pred_train, actual_train)
        #quantile_loss_test = self.calculate_quantile_loss(pred_test, actual_test)
        #self.metrics.update({f'Quantile_Loss_Train_{k}': v for k, v in quantile_loss_train.items()})
        #self.metrics.update({f'Quantile_Loss_Test_{k}': v for k, v in quantile_loss_test.items()})
        
        # Statistical Tests
        #ks_stat_train, ks_p_train = self.calculate_ks_test(pred_train, actual_train)
        #ks_stat_test, ks_p_test = self.calculate_ks_test(pred_test, actual_test)
        #self.metrics['KS_Statistic_Train'] = ks_stat_train
        #self.metrics['KS_Statistic_Test'] = ks_stat_test
        #self.metrics['KS_p_value_Train'] = ks_p_train
        #self.metrics['KS_p_value_Test'] = ks_p_test
        
        #self.metrics['JS_Divergence_Train'] = self.calculate_js_divergence(pred_train, actual_train)
        #self.metrics['JS_Divergence_Test'] = self.calculate_js_divergence(pred_test, actual_test)
        
        # 5. Temporal Metrics
        #self.metrics['Temporal_Correlation_Train'] = self.calculate_temporal_correlation(pred_train, actual_train)
        #self.metrics['Temporal_Correlation_Test'] = self.calculate_temporal_correlation(pred_test, actual_test)
        
        # Store additional data for visualization
        self.evaluation_data = {
            'predictions_train': pred_train,
            'actuals_train': actual_train,
            'predictions_test': pred_test,
            'actuals_test': actual_test,
            'fpr_curve_train': fpr_curve_train,
            'tpr_curve_train': tpr_curve_train,
            'fpr_curve_test': fpr_curve_test,
            'tpr_curve_test': tpr_curve_test,
        }
        
        return self.metrics

    def generate_report(self):
        """Generate comprehensive text report including new metrics"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("B-MTGNN MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.model_path}")
        report_lines.append(f"Data: {self.data_file}")
        report_lines.append("")
        
        # Model Configuration
        report_lines.append("MODEL CONFIGURATION")
        report_lines.append("-" * 40)
        for key, value in self.config.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Calculate overall scores
        avg_msmape = (self.metrics['M-SMAPE_Train'] + self.metrics['M-SMAPE_Test']) / 2
        avg_mae = (self.metrics['MAE_Train'] + self.metrics['MAE_Test']) / 2
        avg_mape = (self.metrics['MAPE_Train'] + self.metrics['MAPE_Test']) / 2
        avg_rmse = (self.metrics['RMSE_Train'] + self.metrics['RMSE_Test']) / 2
        avg_rae = (self.metrics['RAE_Train'] + self.metrics['RAE_Test']) / 2
        avg_rrse = (self.metrics['RRSE_Train'] + self.metrics['RRSE_Test']) / 2
        avg_auc = (self.metrics['AUC_ROC_Train'] + self.metrics['AUC_ROC_Test']) / 2
        
        report_lines.append(f"Average M-SMAPE: {avg_msmape:.2f}%")
        report_lines.append(f"Average MAE: {avg_mae:.4f}")
        report_lines.append(f"Average MAPE: {avg_mape:.2f}%")
        report_lines.append(f"Average RMSE: {avg_rmse:.4f}")
        report_lines.append(f"Average RAE: {avg_rae:.4f}")
        report_lines.append(f"Average RRSE: {avg_rrse:.4f}")
        report_lines.append(f"Average AUC-ROC: {avg_auc:.3f}")
        report_lines.append("")
        
        # Detailed Metrics
        report_lines.append("DETAILED METRICS")
        report_lines.append("-" * 40)
        
        # Group metrics by category
        accuracy_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['SMAPE', 'MAE', 'MAPE', 'RMSE', 'RAE', 'RRSE', 'Directional'])}
        detection_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['AUC', 'Precision'])}
        #temporal_metrics = {k: v for k, v in self.metrics.items() if 'Temporal' in k}
        #robustness_metrics = {k: v for k, v in self.metrics.items() if any(x in k for x in ['KS', 'JS'])}
        #quantile_metrics = {k: v for k, v in self.metrics.items() if 'Quantile' in k}
        
        report_lines.append("Accuracy Metrics:")
        for metric, value in sorted(accuracy_metrics.items()):
            if 'SMAPE' in metric or 'MAPE' in metric:
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.2f}%")
            elif any(x in metric for x in ['MAE', 'RMSE', 'RAE', 'RRSE']):
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        """
        report_lines.append("\nDetection Quality Metrics:")
        for metric, value in sorted(detection_metrics.items()):
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nTemporal Performance Metrics:")
        for metric, value in sorted(temporal_metrics.items()):
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nRobustness and Distribution Metrics:")
        for metric, value in sorted(robustness_metrics.items()):
            if 'p_value' in metric:
                report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nUncertainty Quantification (Quantile Loss):") 
        
        """
        #for metric, value in sorted(quantile_metrics.items()):
        #    if value == float('inf'):
        #        report_lines.append(f"  {metric}: N/A (no predictions)")
        #    else:
        #        report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.append("")
        
        # Performance Assessment
        report_lines.append("PERFORMANCE ASSESSMENT")
        report_lines.append("-" * 40)
        
        # Assess model performance based on new metrics
        if avg_auc > 0.8:
            auc_assessment = "EXCELLENT"
        elif avg_auc > 0.7:
            auc_assessment = "GOOD"
        elif avg_auc > 0.6:
            auc_assessment = "FAIR"
        else:
            auc_assessment = "POOR"
        
        if avg_rmse == float('inf'):
            rmse_assessment = "N/A"
        elif avg_rmse < 0.1:
            rmse_assessment = "EXCELLENT"
        elif avg_rmse < 0.2:
            rmse_assessment = "GOOD"
        elif avg_rmse < 0.3:
            rmse_assessment = "FAIR"
        else:
            rmse_assessment = "POOR"
        
        if avg_msmape == float('inf'):
            smape_assessment = "N/A"
        elif avg_msmape < 10:
            smape_assessment = "EXCELLENT"
        elif avg_msmape < 20:
            smape_assessment = "GOOD"
        elif avg_msmape < 30:
            smape_assessment = "FAIR"
        else:
            smape_assessment = "POOR"
        
        report_lines.append(f"Detection Quality: {auc_assessment} (AUC-ROC: {avg_auc:.3f})")
        if avg_rmse == float('inf'):
            report_lines.append(f"RMSE Performance: N/A (no predictions generated)")
        else:
            report_lines.append(f"RMSE Performance: {rmse_assessment} (RMSE: {avg_rmse:.4f})")
        
        if avg_msmape == float('inf'):
            report_lines.append(f"Forecast Accuracy: N/A (no predictions generated)")
        else:
            report_lines.append(f"Forecast Accuracy: {smape_assessment} (M-SMAPE: {avg_msmape:.2f}%)")
        
        # RAE and RRSE assessment (lower is better)
        if avg_rae == float('inf'):
            rae_assessment = "N/A"
        elif avg_rae < 0.5:
            rae_assessment = "EXCELLENT"
        elif avg_rae < 0.8:
            rae_assessment = "GOOD"
        elif avg_rae < 1.0:
            rae_assessment = "FAIR"
        else:
            rae_assessment = "POOR"
            
        if avg_rrse == float('inf'):
            rrse_assessment = "N/A"
        elif avg_rrse < 0.5:
            rrse_assessment = "EXCELLENT"
        elif avg_rrse < 0.8:
            rrse_assessment = "GOOD"
        elif avg_rrse < 1.0:
            rrse_assessment = "FAIR"
        else:
            rrse_assessment = "POOR"
        
        report_lines.append(f"Relative Accuracy (RAE): {rae_assessment} (RAE: {avg_rae:.4f})")
        report_lines.append(f"Relative Squared Error (RRSE): {rrse_assessment} (RRSE: {avg_rrse:.4f})")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        """if self.metrics['FPR_Test'] > 0.1:
            report_lines.append("HIGH FALSE POSITIVE RATE: Consider adjusting detection threshold")
        if self.metrics['Directional_Accuracy_Test'] < 0.6:
            report_lines.append("POOR TREND PREDICTION: Focus on temporal feature engineering")
        """
        # Recommendations based on new metrics
        if self.metrics['RRSE_Test'] > 1.0:
            report_lines.append("HIGH RELATIVE SQUARED ERROR: Model variance is high compared to baseline")
        if self.metrics['RAE_Test'] > 1.0:
            report_lines.append("HIGH RELATIVE ABSOLUTE ERROR: Model performs worse than mean predictor")
        if self.metrics['RMSE_Test'] > 0.3:
            report_lines.append("HIGH RMSE: Consider improving model accuracy or data quality")
        
        if self.metrics['AUC_ROC_Test'] > 0.7 and self.metrics['FPR_Test'] < 0.2:
            report_lines.append("REASONABLE DETECTION CAPABILITY: Model shows promise")
        
        if self.metrics['RRSE_Test'] < 0.8 and self.metrics['RAE_Test'] < 0.8:
            report_lines.append("GOOD RELATIVE PERFORMANCE: Model outperforms baseline predictors")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        report_path = "b_mtgnn_baseline_evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path

def main():
    """Main execution function"""
    # Configuration for B-MTGNN model
    model_path = "Dissertation/BMTGNN Baseline/model/Bayesian/o_model.pt"  # Path to model
    data_file = "Dissertation/BMTGNN Baseline/data/sm_data_g.csv"  # Path to dataset
    
    # Can customize the config based on your trained model's hyperparameters
    config = {
        'gcn_true': True,
        'buildA_true': True,
        'num_nodes': 645,
        'seq_length': 12,
        'forecast_horizon': 12,
        'dropout': 0.3,
        'subgraph_size': 20,
        'node_dim': 40,
        'dilation_exponential': 1,
        'conv_channels': 32,
        'residual_channels': 32,
        'skip_channels': 64,
        'end_channels': 128,
        'layers': 3,
        'propalpha': 0.05,
        'tanhalpha': 3,
        'gcn_depth': 2,
        'normalize': 2,
        'batch_size': 4
    }
    
    # Initialize evaluator
    evaluator = BMTGNNEvaluator(
        model_path=model_path,
        data_file=data_file,
        config=config
    )
    
    # Run comprehensive evaluation
    metrics = evaluator.run_comprehensive_evaluation()
    
    # Generate visualizations
    evaluator.plot_detailed_analysis()
    
    # Generate text report
    report_path = evaluator.generate_report()
    
    print("\nEvaluation completed successfully!")
    print(f"Key metrics calculated: {len(metrics)}")
    print(f"Report generated: {report_path}")
    
    # Print key metrics to console
    print("\nKEY METRICS SUMMARY:")
    print(f"M-SMAPE Train: {metrics.get('M-SMAPE_Train', 'N/A'):.2f}%")
    print(f"M-SMAPE Test: {metrics.get('M-SMAPE_Test', 'N/A'):.2f}%")
    print(f"MAE Train: {metrics.get('MAE_Train', 'N/A'):.4f}")
    print(f"MAE Test: {metrics.get('MAE_Test', 'N/A'):.4f}")
    print(f"MAPE Train: {metrics.get('MAPE_Train', 'N/A'):.2f}%")
    print(f"MAPE Test: {metrics.get('MAPE_Test', 'N/A'):.2f}%")
    print(f"RMSE Train: {metrics.get('RMSE_Train', 'N/A'):.4f}")
    print(f"RMSE Test: {metrics.get('RMSE_Test', 'N/A'):.4f}")
    print(f"RAE Train: {metrics.get('RAE_Train', 'N/A'):.4f}")
    print(f"RAE Test: {metrics.get('RAE_Test', 'N/A'):.4f}")
    print(f"RRSE Train: {metrics.get('RRSE_Train', 'N/A'):.4f}")
    print(f"RRSE Test: {metrics.get('RRSE_Test', 'N/A'):.4f}")
    print(f"AUC-ROC Train: {metrics.get('AUC_ROC_Train', 'N/A'):.3f}")
    print(f"AUC-ROC Test: {metrics.get('AUC_ROC_Test', 'N/A'):.3f}")
    #print(f"Directional Accuracy Test: {metrics.get('Directional_Accuracy_Test', 'N/A'):.3f}")

if __name__ == "__main__":
    main()