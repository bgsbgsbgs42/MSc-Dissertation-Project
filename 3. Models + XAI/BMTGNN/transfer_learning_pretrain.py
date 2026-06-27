import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import csv
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.autograd import Variable
from datetime import datetime
import numbers
import glob

plt.rcParams['savefig.dpi'] = 1200

# Create necessary directories
os.makedirs('model/Validation', exist_ok=True)
os.makedirs('model/Testing', exist_ok=True)
os.makedirs('transfer_learning_pretrain', exist_ok=True)

#  DATA LOADER 
class DataLoaderS(object):
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1):
        self.P = window
        self.h = horizon
        self.data_file = file_name
        self.device = device
        self.normalize = normalize
        self.out_len = out
        
        self.read_data()
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m)

        self.scale = self.scale.to(self.device)
        self.scale = Variable(self.scale)

        self.rse = self.normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.adj = self.build_predefined_adj()

    def read_data(self):
        print(f"Loading data from: {self.data_file}")
        try:
            # Read with headers to get node names
            self.data = pd.read_csv(self.data_file, header=0)
            self.feature_names = self.data.columns.tolist()
            print(f"Loaded {len(self.feature_names)} nodes from header")
        except Exception as e:
            print(f"Error reading with headers: {e}, trying without headers")
            self.data = pd.read_csv(self.data_file, header=None)
            self.feature_names = [f"Node_{i}" for i in range(self.data.shape[1])]
            
        self.rawdat = self.data.values
        print(f"Raw data shape: {self.rawdat.shape}")

        self.n, self.m = self.rawdat.shape
        self.col = self.feature_names

        # Create date range from July 2011 to December 2024
        self.dates = []
        current_date = datetime(2011, 7, 1)
        end_date = datetime(2024, 12, 1)
        while current_date <= end_date:
            self.dates.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        print(f"Date range: {self.dates[0].strftime('%Y-%m')} to {self.dates[-1].strftime('%Y-%m')}")
        print(f"Total months: {len(self.dates)}")

    def normal_std(self, x):
        return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

    def _normalized(self, normalize):
        self.dat = np.zeros(self.rawdat.shape)
        
        if normalize == 0:
            self.dat = self.rawdat
        elif normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
        elif normalize == 2:
            self.scale = np.ones(self.m)
            for i in range(self.m):
                # Add small epsilon to avoid division by zero
                max_val = np.max(np.abs(self.rawdat[:, i]))
                if max_val == 0 or np.isnan(max_val):
                    self.scale[i] = 1.0  # Avoid division by zero
                    self.dat[:, i] = self.rawdat[:, i]
                else:
                    self.scale[i] = max_val
                    self.dat[:, i] = self.rawdat[:, i] / max_val

    def _split(self, train, valid, test):
        # Ensure we have enough data for the window size and output length
        min_required = self.P + self.h + self.out_len - 1
        if self.n < min_required:
            raise ValueError(f"Not enough data points. Need at least {min_required}, but only have {self.n}")
        
        # Adjust split points to ensure positive dimensions
        train_end = min(train, self.n - self.out_len - 10)  # Leave room for validation
        valid_end = min(valid, self.n - self.out_len)
        
        print(f"Splitting data: train={self.P+self.h-1} to {train_end}, valid={train_end} to {valid_end}, test={valid_end} to {self.n}")
        
        train_set = range(self.P + self.h - 1, train_end)
        valid_set = range(train_end, valid_end)
        test_set = range(valid_end, self.n)
        
        print(f"Train set size: {len(train_set)}, Valid set size: {len(valid_set)}, Test set size: {len(test_set)}")
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        
        # For testing window, use the last part of the data
        test_window_start = max(0, self.n - (36 + self.P))
        self.test_window = torch.from_numpy(self.dat[test_window_start:, :])
        print(f"Test window shape: {self.test_window.shape}")

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        if n <= self.out_len:
            # Not enough data points for this split
            print(f"Warning: Not enough data points in split. Have {n}, need more than {self.out_len}")
            # Create empty tensors
            X = torch.zeros((0, self.P, self.m))
            Y = torch.zeros((0, self.out_len, self.m))
            return [X, Y]
        
        X = torch.zeros((n - self.out_len, self.P, self.m))
        Y = torch.zeros((n - self.out_len, self.out_len, self.m))

        for i in range(n - self.out_len):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i] + self.out_len, :])
            
        print(f"Batchified shape: X={X.shape}, Y={Y.shape}")
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if length == 0:
            # Return empty generator if no data
            return
            yield
        
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    def build_predefined_adj(self):
        graph = defaultdict(list)
        try:
            with open('data/graph.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    key_node = row[0]
                    adjacent_nodes = [node for node in row[1:] if node]
                    graph[key_node].extend(adjacent_nodes)
            print('Graph loaded with', len(graph), 'attacks...')
        except FileNotFoundError:
            print('Graph file not found, using identity matrix')
            return torch.eye(len(self.col))

        # Create adjacency matrix
        adj = torch.zeros((len(self.col), len(self.col)))
        connected_nodes = 0
        
        for i in range(adj.shape[0]):
            node_name = self.col[i]
            if node_name in graph:
                for j in range(adj.shape[1]):
                    if self.col[j] in graph[node_name]:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        connected_nodes += 1
        
        print(f'Adjacency created with {connected_nodes} connections...')
        return adj

#  JSON SERIALIZATION HELPERS 
def convert_to_serializable(obj):
    """Convert PyTorch tensors and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        # Convert tensor to list or scalar
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, defaultdict):
        return dict(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj

#  NEURAL NETWORK LAYERS 
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
        # Add small epsilon to avoid division by zero
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

#  MAIN MODEL 
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

#  OPTIMIZER 
class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = list(params)
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._makeOptimizer()

    def step(self):
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        for param in self.params:
            if param.grad is not None:
                grad_norm += math.pow(param.grad.data.norm(), 2)
        
        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.clip / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if param.grad is not None and shrinkage < 1:
                param.grad.data.mul_(shrinkage)
                
        self.optimizer.step()
        return grad_norm

#  TRANSFER LEARNING COMPONENTS 

class DomainPreTrainer:
    """Handles pre-training on a single domain"""
    
    def __init__(self, domain_name, model_class, base_config, device):
        self.domain_name = domain_name
        self.model_class = model_class
        self.base_config = base_config.copy()
        self.device = device
        self.best_state = None
        self.training_history = []
    
    def load_and_prepare_data(self, data_path, sequence_length=12, forecast_horizon=6):
        """Load domain data and create sequences"""
        print(f"\nLoading {self.domain_name} data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            
            # Remove date/timestamp columns and keep only numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_cols].values
            
            print(f"  Raw shape: {data.shape}")
            print(f"  Features: {len(numeric_cols)}")
            
            # Normalize data
            scale = np.ones(data.shape[1])
            for i in range(data.shape[1]):
                max_val = np.max(np.abs(data[:, i]))
                if max_val == 0 or np.isnan(max_val):
                    scale[i] = 1.0
                else:
                    scale[i] = max_val
            normalized_data = data / scale
            
            # Create sequences
            n = len(normalized_data) - sequence_length - forecast_horizon + 1
            if n <= 0:
                print(f"  WARNING: Not enough data to create sequences!")
                return None
                
            X = np.zeros((n, sequence_length, data.shape[1]))
            Y = np.zeros((n, forecast_horizon, data.shape[1]))
            
            for i in range(n):
                X[i] = normalized_data[i:i+sequence_length]
                Y[i] = normalized_data[i+sequence_length:i+sequence_length+forecast_horizon]
            
            print(f"  Created {len(X)} sequences")
            print(f"  X shape: {X.shape}, y shape: {Y.shape}")
            
            # Split into train/val
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = Y[:train_size], Y[train_size:]
            
            return {
                'X_train': torch.FloatTensor(X_train),
                'y_train': torch.FloatTensor(y_train),
                'X_val': torch.FloatTensor(X_val),
                'y_val': torch.FloatTensor(y_val),
                'num_nodes': data.shape[1],
                'scale': torch.FloatTensor(scale),
                'original_data': data
            }
            
        except Exception as e:
            print(f"  ERROR loading {data_path}: {e}")
            return None
    
    def pretrain(self, data_dict, epochs=50, batch_size=8, learning_rate=0.001):
        """Pre-train model on this domain"""
        if data_dict is None:
            print(f"  SKIPPING {self.domain_name} - no valid data")
            return None
            
        print(f"\n{'='*80}")
        print(f"PRE-TRAINING ON {self.domain_name.upper()}")
        print(f"{'='*80}")
        
        # Adjust config for this domain
        config = self.base_config.copy()
        config['num_nodes'] = data_dict['num_nodes']
        config['seq_length'] = data_dict['X_train'].shape[1]
        config['out_dim'] = data_dict['y_train'].shape[1]
        
        print(f"  Model config: num_nodes={config['num_nodes']}, seq_length={config['seq_length']}, out_dim={config['out_dim']}")
        
        # Create model
        model = self.model_class(
            gcn_true=config['gcn_true'],
            buildA_true=config['buildA_true'], 
            gcn_depth=config['gcn_depth'],
            num_nodes=config['num_nodes'],
            device=self.device,
            predefined_A=None,
            dropout=config['dropout'],
            subgraph_size=config['subgraph_size'],
            node_dim=config['node_dim'],
            dilation_exponential=config['dilation_exponential'],
            conv_channels=config['conv_channels'],
            residual_channels=config['residual_channels'],
            skip_channels=config['skip_channels'],
            end_channels=config['end_channels'],
            seq_length=config['seq_length'],
            in_dim=config['in_dim'],
            out_dim=config['out_dim'],
            layers=config['layers'],
            propalpha=config['propalpha'],
            tanhalpha=config['tanhalpha']
        ).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
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
                
                # Reshape for model
                X_batch = X_batch.unsqueeze(1).transpose(2, 3)
                
                optimizer.zero_grad()
                output = model(X_batch)
                output = output.squeeze(3)
                
                # Apply scaling
                scale_batch = data_dict['scale'].expand(output.size(0), output.size(1), data_dict['num_nodes'])
                output = output * scale_batch
                y_batch = y_batch * scale_batch
                
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else train_loss
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val = data_dict['X_val'].to(self.device)
                y_val = data_dict['y_val'].to(self.device)
                
                X_val = X_val.unsqueeze(1).transpose(2, 3)
                val_output = model(X_val)
                val_output = val_output.squeeze(3)
                
                # Apply scaling
                scale_val = data_dict['scale'].expand(val_output.size(0), val_output.size(1), data_dict['num_nodes'])
                val_output = val_output * scale_val
                y_val = y_val * scale_val
                
                val_loss = criterion(val_output, y_val)
            
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
            
            if patience_counter >= max_patience:
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
        target_model = self.model_class(
            gcn_true=self.target_config['gcn_true'],
            buildA_true=self.target_config['buildA_true'], 
            gcn_depth=self.target_config['gcn_depth'],
            num_nodes=self.target_config['num_nodes'],
            device=self.device,
            predefined_A=self.target_config['predefined_A'],
            dropout=self.target_config['dropout'],
            subgraph_size=self.target_config['subgraph_size'],
            node_dim=self.target_config['node_dim'],
            dilation_exponential=self.target_config['dilation_exponential'],
            conv_channels=self.target_config['conv_channels'],
            residual_channels=self.target_config['residual_channels'],
            skip_channels=self.target_config['skip_channels'],
            end_channels=self.target_config['end_channels'],
            seq_length=self.target_config['seq_length'],
            in_dim=self.target_config['in_dim'],
            out_dim=self.target_config['out_dim'],
            layers=self.target_config['layers'],
            propalpha=self.target_config['propalpha'],
            tanhalpha=self.target_config['tanhalpha']
        ).to(self.device)
        
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
        # robust load
        report = safe_load_state_dict(target_model, target_state, strict=False, verbose=True)
        print(f"Transferred {len(report['transferred'])} keys from {self.pretrained_states[0]['domain']}")
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
        
        # robust load
        report = safe_load_state_dict(target_model, target_state, strict=False, verbose=True)
        print(f"Averaged and transferred {len(report['transferred'])} layers from {len(self.pretrained_states)} domains")
        
        return target_model, transferred

    def _transfer_selective(self, target_model, target_state):
        """Selectively transfer only transformer layers, not domain-specific layers"""
        print("Selectively transferring compatible layers...")
        
        if not self.pretrained_states:
            print("No pre-trained models available for selective transfer")
            return target_model, []
            
        # Transfer only compatible layers (graph construction, convolution layers)
        layers_to_transfer = ['gc', 'filter_convs', 'gate_convs', 'gconv1', 'gconv2', 'norm']
        layers_to_skip = ['start_conv', 'end_conv_1', 'end_conv_2', 'skip0', 'skipE']
        
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
        
        # robust load
        report = safe_load_state_dict(target_model, target_state, strict=False, verbose=True)
        print(f"Selectively transferred {len(report['transferred'])} layers (requested {len(transferred)})")
        
        return target_model, transferred

def safe_load_state_dict(model: nn.Module, src_state: dict, strict: bool = False, verbose: bool = True):
	"""Load a state_dict into model tolerating missing / mismatched keys.
	Copies tensors from src_state into a new state dict matching model.keys().
	Returns a report dict with lists of transferred/mismatched/missing/extra keys.
	"""
	model_state = model.state_dict()
	new_state = {}
	transferred, mismatched, missing, extra = [], [], [], []

	for k_model, v_model in model_state.items():
		if k_model in src_state:
			v_src = src_state[k_model]
			if isinstance(v_src, torch.Tensor) and v_src.shape == v_model.shape:
				new_state[k_model] = v_src
				transferred.append(k_model)
			else:
				# shape mismatch or non-tensor - skip and keep model param
				new_state[k_model] = v_model
				mismatched.append((k_model, getattr(v_src, 'shape', None)))
				if verbose:
					print(f"[WARN] Shape mismatch or incompatible for '{k_model}': model {v_model.shape} vs src {getattr(v_src,'shape',None)} - keeping model param")
		else:
			# missing in source
			new_state[k_model] = v_model
			missing.append(k_model)

	for k_src in src_state.keys():
		if k_src not in model_state:
			extra.append(k_src)

	# Load with non-strict to avoid PyTorch raising if sizes differ in buffers etc.
	model.load_state_dict(new_state, strict=False)

	if verbose:
		print(f"[safe_load_state_dict] transferred={len(transferred)}, mismatched={len(mismatched)}, missing={len(missing)}, extra={len(extra)}")

	return {
		"transferred": transferred,
		"mismatched": mismatched,
		"missing": missing,
		"extra": extra
	}

#  UTILITY FUNCTIONS 
def consistent_name(name):
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
        if len(word) <= 3 or '/' in word or word == 'MITM' or word == 'SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        if i < len(words) - 1:
            result += ' '
    return result

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, avoiding division by zero"""
    if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
        return default
    return numerator / denominator

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='B-MTGNN Transfer Learning Pre-training')
    parser.add_argument('--pretrain_data_dir', type=str, default='./pretrain_data/', help='directory with pre-training CSV files')
    parser.add_argument('--cyber_data', type=str, default='./data/sm_data_g.csv', help='cyber threat data file')
    parser.add_argument('--epochs_pretrain', type=int, default=50, help='epochs for pre-training')
    parser.add_argument('--epochs_finetune', type=int, default=20, help='epochs for initial fine-tuning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--validate_transfer', action='store_true', default=True, help='run transfer learning validation')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Set random seed
    set_random_seed(123)
    
    # Base configuration for pre-training
    base_config = {
        'gcn_true': True,
        'buildA_true': True,
        'gcn_depth': 2,
        'dropout': 0.3,
        'subgraph_size': 20,
        'node_dim': 40,
        'dilation_exponential': 1,
        'conv_channels': 32,
        'residual_channels': 32,
        'skip_channels': 64,
        'end_channels': 128,
        'seq_length': 12,
        'in_dim': 1,
        'out_dim': 6,  # Shorter horizon for pre-training
        'layers': 3,
        'propalpha': 0.05,
        'tanhalpha': 3
    }
    
    # PRE-TRAINING 
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON RELATED DOMAINS")
    print("="*80)
    
    # Find all CSV files in pretrain directory
    pretrain_files = glob.glob(os.path.join(args.pretrain_data_dir, "*.csv"))
    print(f"Found {len(pretrain_files)} pre-training files")
    
    pretrained_models = []
    
    for file_path in pretrain_files:
        domain_name = os.path.splitext(os.path.basename(file_path))[0]
        
        pretrainer = DomainPreTrainer(
            domain_name, gtnet, base_config, device
        )
        
        data = pretrainer.load_and_prepare_data(file_path)
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
    
    # TRANSFER + INITIAL FINE-TUNING
    if pretrained_models:
        print("\n" + "="*80)
        print("PHASE 2: TRANSFER + INITIAL FINE-TUNING")
        print("="*80)
        
        # Load cyber threat data
        print("Loading cyber threat data for fine-tuning...")
        cyber_data_loader = DataLoaderS(args.cyber_data, 0.7, 0.15, device, horizon=1, window=12, normalize=2, out=12)
        
        # Cyber threat model config
        cyber_config = base_config.copy()
        cyber_config['num_nodes'] = cyber_data_loader.m
        cyber_config['out_dim'] = cyber_data_loader.out_len
        cyber_config['predefined_A'] = cyber_data_adj = cyber_data_loader.adj.to(device)
        
        print(f"Cyber threat model config: num_nodes={cyber_config['num_nodes']}, out_dim={cyber_config['out_dim']}")
        
        # Transfer weights
        transfer_manager = TransferManager(
            pretrained_models, cyber_config, gtnet, device
        )
        
        transferred_model, transferred_layers = transfer_manager.create_transferred_model(
            strategy='selective'
        )
        
        # Initial fine-tuning
        print(f"\nInitial fine-tuning for {args.epochs_finetune} epochs...")
        
        # Use the same training setup as original
        if True:  # Using L1 loss as in original
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        
        evaluateL2 = nn.MSELoss(reduction='sum').to(device)
        evaluateL1 = nn.L1Loss(reduction='sum').to(device)
        
        # Use lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(transferred_model.parameters(), lr=0.0001, weight_decay=0.00001)
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(1, args.epochs_finetune + 1):
            # Training
            transferred_model.train()
            total_loss = 0
            n_samples = 0
            
            for X_batch, Y_batch in cyber_data_loader.get_batches(cyber_data_loader.train[0], cyber_data_loader.train[1], batch_size=4, shuffle=True):
                transferred_model.zero_grad()
                X_batch = torch.unsqueeze(X_batch, dim=1)
                X_batch = X_batch.transpose(2, 3)
                
                output = transferred_model(X_batch)
                output = torch.squeeze(output, 3)
                scale = cyber_data_loader.scale.expand(output.size(0), output.size(1), cyber_data_loader.m)
                output = output * scale
                Y_batch = Y_batch * scale
                
                loss = criterion(output, Y_batch)
                loss.backward()
                total_loss += loss.item()
                n_samples += (output.size(0) * output.size(1) * cyber_data_loader.m)
                optimizer.step()
            
            avg_train_loss = total_loss / n_samples if n_samples > 0 else total_loss
            
            # Validation
            transferred_model.eval()
            with torch.no_grad():
                val_outputs = []
                for X_batch, Y_batch in cyber_data_loader.get_batches(cyber_data_loader.valid[0], cyber_data_loader.valid[1], batch_size=4, shuffle=False):
                    X_batch = torch.unsqueeze(X_batch, dim=1)
                    X_batch = X_batch.transpose(2, 3)
                    output = transferred_model(X_batch)
                    output = torch.squeeze(output, 3)
                    scale = cyber_data_loader.scale.expand(output.size(0), output.size(1), cyber_data_loader.m)
                    output = output * scale
                    Y_batch = Y_batch * scale
                    val_outputs.append((output, Y_batch))
                
                # Calculate validation loss
                val_loss = 0
                val_samples = 0
                for output, Y_batch in val_outputs:
                    val_loss += evaluateL2(output, Y_batch).item()
                    val_samples += (output.size(0) * output.size(1) * cyber_data_loader.m)
                
                avg_val_loss = val_loss / val_samples if val_samples > 0 else val_loss
            
            print(f"Fine-tune Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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
        
        # Save transfer learning summary
        transfer_summary = {
            'pretrained_domains': [p['domain'] for p in pretrained_models],
            'transferred_layers': transferred_layers,
            'fine_tune_val_loss': best_val_loss,
            'cyber_data_info': {
                'num_nodes': cyber_data_loader.m,
                'train_samples': len(cyber_data_loader.train[0]),
                'valid_samples': len(cyber_data_loader.valid[0]),
                'test_samples': len(cyber_data_loader.test[0])
            }
        }
        
        with open('transfer_learning_pretrain/transfer_summary.json', 'w') as f:
            json.dump(convert_to_serializable(transfer_summary), f, indent=2)
    
    else:
        print("No pre-trained models available for transfer learning.")
    
    print(f"\nTransfer learning pipeline completed.")
    print(f"Results saved in: transfer_learning_pretrain/")

if __name__ == "__main__":
    main()