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

plt.rcParams['savefig.dpi'] = 1200

# Create necessary directories
os.makedirs('model/Validation', exist_ok=True)
os.makedirs('model/Testing', exist_ok=True)

# DATA LOADER 
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
        
        # If test targets exist use them for rse/rae; otherwise set safe defaults
        if hasattr(self, 'test') and isinstance(self.test, (list, tuple)) and len(self.test) > 1 and getattr(self.test[1], 'numel', lambda: 0)() > 0:
            tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m)
            self.rse = self.normal_std(tmp)
            try:
                self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
            except Exception:
                self.rae = 0.0
        else:
            # No test data available -> safe defaults
            self.rse = 0.0
            self.rae = 0.0

        self.scale = self.scale.to(self.device)
        self.scale = Variable(self.scale)

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
        # Robust std estimator: handle empty or singleton inputs to avoid division by zero.
        try:
            n = len(x)
        except Exception:
            # Fallback for scalars / unknown types
            arr = np.asarray(x)
            n = arr.size if hasattr(arr, 'size') else 0

        if n <= 1:
            return 0.0

        # x may be a numpy array or torch tensor; get the standard deviation as a float
        if hasattr(x, 'std'):
            std_val = x.std()
        else:
            std_val = np.std(np.asarray(x))

        if hasattr(std_val, 'item'):
            std_val = std_val.item()

        correction = math.sqrt((n - 1.) / float(n))
        return float(std_val * correction)

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
            with open('Dissertation/Bayesian MTGNN/data/graph.csv', 'r') as f:
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

# JSON SERIALIZATION HELPERS 
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

def save_metrics_1d(predict, test, title, type):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff + epsilon)
    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    
    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r + epsilon)
    
    rrse = safe_divide(root_sum_squared, root_sum_squared_r)
    
    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = safe_divide(sum_absolute_diff, sum_absolute_r)

    title = title.replace('/', '_')
    with open(f'model/{type}/{title}_{type}.txt', "w") as f:
        f.write(f'rse:{rrse}\n')
        f.write(f'rae:{rae}\n')
        f.close()

def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    M = []
    
    # Create date range from July 2011 to December 2024
    for year in range(11, 25):  # 2011 to 2024
        for month in months:
            if year == 11 and month not in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                continue
            M.append(month + '-' + str(year))
    
    M2 = []
    p = []
    
    if type == 'Testing':
        # For testing: use the last part of the timeline (last 36 months)
        M = M[-len(predicted):]
        for index, value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index + 1)
    else:
        # For validation: October 2016 to September 2019
        # Find the start index for Oct-16
        start_idx = None
        for i, date_str in enumerate(M):
            if date_str == 'Oct-16':
                start_idx = i
                break
        
        if start_idx is not None:
            # Get 36 months from Oct-16 to Sep-19
            M = M[start_idx:start_idx + len(predicted)]
            for index, value in enumerate(M):
                if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                    M2.append(M[index])
                    p.append(index + 1)
        else:
            # Fallback: use default labeling
            for index in range(len(predicted)):
                if index % 3 == 0:  # Every 3 months
                    M2.append(f'Month {index+1}')
                    p.append(index + 1)

    x = range(1, len(predicted) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted, '--', color='purple', label='Predicted', linewidth=2)
    
    # Ensure confidence intervals are valid
    if confidence_95 is not None and not torch.isnan(confidence_95).any():
        lower_bound = predicted - confidence_95.numpy()
        upper_bound = predicted + confidence_95.numpy()
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.3, color='pink', label='95% Confidence')
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True, alpha=0.3)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    
    if M2:  # Only set custom ticks if we have labels
        plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=11)
    else:
        plt.xticks(fontsize=11)
    
    plt.yticks(fontsize=11)
    title = title.replace('/', '_')
    plt.savefig(f'model/{type}/{title}_{type}.png', bbox_inches="tight", dpi=300)
    plt.savefig(f'model/{type}/{title}_{type}.pdf', bbox_inches="tight", format='pdf')
    plt.close()

def safe_smape(yTrue, yPred):
    """Safe symmetric MAPE calculation that handles zeros and NaNs"""
    total = 0
    count = 0
    epsilon = 1e-8  # Small epsilon to avoid division by zero
    
    for i in range(len(yTrue)):
        denominator = abs(yTrue[i]) + abs(yPred[i]) + epsilon
        if not (math.isnan(yTrue[i]) or math.isnan(yPred[i]) or math.isinf(denominator)):
            total += abs(yTrue[i] - yPred[i]) / denominator
            count += 1
    
    if count == 0:
        return float('nan')
    
    return total / count

# TRAINING AND EVALUATION 
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # Check if we have any training data
    if len(X) == 0:
        print("No training data available!")
        return float('inf')

    for X_batch, Y_batch in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X_batch = torch.unsqueeze(X_batch, dim=1)
        X_batch = X_batch.transpose(2, 3)
        
        # Use all nodes (no splitting for simplicity with small dataset)
        output = model(X_batch)
        output = torch.squeeze(output, 3)
        scale = data.scale.expand(output.size(0), output.size(1), data.m)
        output = output * scale
        Y_batch = Y_batch * scale
        
        loss = criterion(output, Y_batch)
        loss.backward()
        total_loss += loss.item()
        n_samples += (output.size(0) * output.size(1) * data.m)
        grad_norm = optim.step()

        if iter % 10 == 0:  # Reduced frequency for cleaner output
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * output.size(1) * data.m)))
        iter += 1
    
    if n_samples == 0:
        return float('inf')
    return total_loss / n_samples

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0

    # Check if we have any validation data
    if len(X) == 0:
        print("No validation data available!")
        return float('inf'), float('inf'), 0, float('nan')

    for X_batch, Y_batch in data.get_batches(X, Y, batch_size, False):
        X_batch = torch.unsqueeze(X_batch, dim=1)
        X_batch = X_batch.transpose(2, 3)
        num_runs = 10
        outputs = []
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X_batch)
                output = torch.squeeze(output)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0).unsqueeze(0)
                elif len(output.shape) == 2:
                    output = output.unsqueeze(0)
                outputs.append(output)
        
        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)
        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs, dtype=torch.float))
        output = mean
        
        scale = data.scale.expand(Y_batch.size(0), Y_batch.size(1), data.m)
        output = output * scale
        Y_batch = Y_batch * scale
        var = var * scale
        confidence = confidence * scale

        if predict is None:
            predict = output
            test = Y_batch
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, output), dim=0)
            test = torch.cat((test, Y_batch), dim=0)
            variance = torch.cat((variance, var), dim=0)
            confidence_95 = torch.cat((confidence_95, confidence), dim=0)

        total_loss += evaluateL2(output, Y_batch).item()
        total_loss_l1 += evaluateL1(output, Y_batch).item()
        n_samples += (output.size(0) * output.size(1) * data.m)
        sum_squared_diff += torch.sum(torch.pow(Y_batch - output, 2))
        sum_absolute_diff += torch.sum(torch.abs(Y_batch - output))

    # Calculate metrics with safe division
    epsilon = 1e-8
    root_sum_squared = math.sqrt(sum_squared_diff + epsilon)
    test_s = test
    mean_all = torch.mean(test_s, dim=(0, 1))
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r + epsilon)
    rrse = safe_divide(root_sum_squared, root_sum_squared_r)
    
    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = safe_divide(sum_absolute_diff, sum_absolute_r)

    predict_np = predict.data.cpu().numpy()
    Ytest_np = test.data.cpu().numpy()
    
    # Calculate correlation safely
    correlation = 0
    valid_correlations = 0
    for node in range(min(10, data.m)):  # Limit to first 10 nodes for efficiency
        pred_node = predict_np[:, :, node].flatten()
        test_node = Ytest_np[:, :, node].flatten()
        
        # Remove NaNs and check for constant values
        mask = ~(np.isnan(pred_node) | np.isnan(test_node))
        pred_clean = pred_node[mask]
        test_clean = test_node[mask]
        
        if len(pred_clean) > 1 and len(test_clean) > 1:
            if np.std(pred_clean) > epsilon and np.std(test_clean) > epsilon:
                corr = np.corrcoef(pred_clean, test_clean)[0, 1]
                if not np.isnan(corr):
                    correlation += corr
                    valid_correlations += 1
    
    correlation = correlation / valid_correlations if valid_correlations > 0 else 0

    # Calculate SMAPE safely
    smape = 0
    valid_smape = 0
    for batch in range(min(5, predict_np.shape[0])):  # Limit batches for efficiency
        for node in range(min(10, predict_np.shape[2])):  # Limit nodes for efficiency
            pred_vals = predict_np[batch, :, node]
            test_vals = Ytest_np[batch, :, node]
            node_smape = safe_smape(test_vals, pred_vals)
            if not math.isnan(node_smape):
                smape += node_smape
                valid_smape += 1
    
    smape = smape / valid_smape if valid_smape > 0 else float('nan')

    if is_plot and predict_np.shape[0] > 0:
        for col in range(min(3, data.m)):  # Plot first 3 nodes to avoid too many plots
            node_name = data.col[col].replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            
            # Use the last batch for plotting
            last_batch = predict_np.shape[0] - 1
            pred_1d = predict_np[last_batch, :, col]
            test_1d = Ytest_np[last_batch, :, col]
            var_1d = variance[last_batch, :, col] if variance is not None else None
            conf_1d = confidence_95[last_batch, :, col] if confidence_95 is not None else None
            
            save_metrics_1d(torch.from_numpy(pred_1d), torch.from_numpy(test_1d), node_name, 'Validation')
            plot_predicted_actual(pred_1d, test_1d, node_name, 'Validation', var_1d, conf_1d)
    
    return rrse, rae, correlation, smape

def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    r = random.randint(0, data.m - 1)
    print('testing r=', str(r))
    
    scale = data.scale.expand(test_window.size(0), data.m)
    x_input = test_window[0:n_input, :].clone()

    # Adjust steps based on available data
    step_size = min(data.out_len, test_window.shape[0] - n_input)
    if step_size <= 0:
        print("Not enough data for sliding window evaluation!")
        return float('inf'), float('inf'), 0, float('nan')

    for i in range(n_input, test_window.shape[0], step_size):
        end_idx = min(i + data.out_len, test_window.shape[0])
        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        X = X.to(torch.float)
        y_true = test_window[i:end_idx, :].clone()
        
        num_runs = 10
        outputs = []
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                # Handle different output shapes
                if len(output.shape) == 4:
                    y_pred = output[-1, :, :, -1].clone()
                else:
                    y_pred = output.clone()
                
                if y_pred.shape[0] > y_true.shape[0]:
                    y_pred = y_pred[:-(y_pred.shape[0] - y_true.shape[0]), ]
                outputs.append(y_pred)
        
        outputs = torch.stack(outputs)
        y_pred_mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)
        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs, dtype=torch.float))

        # Update input for next iteration
        if data.P <= data.out_len:
            x_input = y_pred_mean[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[-(data.P - data.out_len):, :].clone(), y_pred_mean.clone()], dim=0)

        if predict is None:
            predict = y_pred_mean
            test = y_true
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, y_pred_mean), dim=0)
            test = torch.cat((test, y_true), dim=0)
            variance = torch.cat((variance, var), dim=0)
            confidence_95 = torch.cat((confidence_95, confidence), dim=0)

    # Apply scaling
    if predict is not None and test is not None:
        scale_test = data.scale.expand(test.size(0), data.m)
        predict = predict * scale_test
        test = test * scale_test
        if variance is not None:
            variance = variance * scale_test
        if confidence_95 is not None:
            confidence_95 = confidence_95 * scale_test

        # Calculate metrics with safe division
        epsilon = 1e-8
        sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
        sum_absolute_diff = torch.sum(torch.abs(test - predict))
        
        root_sum_squared = math.sqrt(sum_squared_diff + epsilon)
        test_s = test
        mean_all = torch.mean(test_s, dim=0)
        diff_r = test_s - mean_all.expand(test_s.size(0), data.m)
        sum_squared_r = torch.sum(torch.pow(diff_r, 2))
        root_sum_squared_r = math.sqrt(sum_squared_r + epsilon)
        
        rrse = safe_divide(root_sum_squared, root_sum_squared_r)
        sum_absolute_r = torch.sum(torch.abs(diff_r))
        rae = safe_divide(sum_absolute_diff, sum_absolute_r)

        predict_np = predict.data.cpu().numpy()
        Ytest_np = test.data.cpu().numpy()
        
        # Calculate correlation safely
        correlation = 0
        valid_correlations = 0
        for node in range(min(10, data.m)):  # Limit to first 10 nodes for efficiency
            pred_node = predict_np[:, node]
            test_node = Ytest_np[:, node]
            
            # Remove NaNs and check for constant values
            mask = ~(np.isnan(pred_node) | np.isnan(test_node))
            pred_clean = pred_node[mask]
            test_clean = test_node[mask]
            
            if len(pred_clean) > 1 and len(test_clean) > 1:
                if np.std(pred_clean) > epsilon and np.std(test_clean) > epsilon:
                    corr = np.corrcoef(pred_clean, test_clean)[0, 1]
                    if not np.isnan(corr):
                        correlation += corr
                        valid_correlations += 1
        
        correlation = correlation / valid_correlations if valid_correlations > 0 else 0

        # Calculate SMAPE safely
        smape = 0
        valid_smape = 0
        for node in range(min(10, predict_np.shape[1])):  # Limit nodes for efficiency
            pred_vals = predict_np[:, node]
            test_vals = Ytest_np[:, node]
            node_smape = safe_smape(test_vals, pred_vals)
            if not math.isnan(node_smape):
                smape += node_smape
                valid_smape += 1
        
        smape = smape / valid_smape if valid_smape > 0 else float('nan')

        if is_plot:
            for col in range(min(645, data.m)):  # Plot ALL nodes
                node_name = data.col[col].replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
                node_name = consistent_name(node_name)
                
                pred_1d = predict_np[:, col]
                test_1d = Ytest_np[:, col]
                var_1d = variance[:, col] if variance is not None else None
                conf_1d = confidence_95[:, col] if confidence_95 is not None else None
                
                save_metrics_1d(torch.from_numpy(pred_1d), torch.from_numpy(test_1d), node_name, 'Testing')
                plot_predicted_actual(pred_1d, test_1d, node_name, 'Testing', var_1d, conf_1d)
    else:
        print("No predictions generated in sliding window evaluation!")
        rrse, rae, correlation, smape = float('inf'), float('inf'), 0, float('nan')

    return rrse, rae, correlation, smape

# MAIN HYPERPARAMETER OPTIMIZATION 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='B-MTGNN Hyperparameter Optimization with Transfer Learning')
    parser.add_argument('--transferred_model', type=str, default='Dissertation/Bayesian MTGNN/transfer_learning_pretrain/transferred_model_for_hp_search.pt', help='path to transferred model for initialization')
    parser.add_argument('--use_transfer', type=bool, default=True, help='whether to use transferred model initialization')
    parser.add_argument('--data', type=str, default='Dissertation/Bayesian MTGNN/data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
    parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')
    parser.add_argument('--num_nodes', type=int, default=645, help='number of nodes/variables')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=6, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--step_size', type=int, default=100, help='step_size')
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    
    global args, device
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Load transferred model if specified
    transferred_config = None
    if args.use_transfer and os.path.exists(args.transferred_model):
        print(f"Loading transferred model from {args.transferred_model}")
        transferred_checkpoint = torch.load(args.transferred_model, map_location=device)
        transferred_config = transferred_checkpoint['config']
        print(f"Loaded model pre-trained on: {transferred_checkpoint.get('pretrained_domains', ['unknown'])}")
        print(f"Transferred layers: {len(transferred_checkpoint.get('transferred_layers', []))}")
    
    # Hyperparameter search space 
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]
    
    # Number of random search iterations
    n_iterations = 60
    
    set_random_seed(123)
    
    best_val = float('inf')
    best_rse = float('inf')
    best_rae = float('inf')
    best_corr = -float('inf')
    best_smape = float('inf')
    best_test_rse = float('inf')
    best_test_corr = -float('inf')
    best_hp = []
    
    optimization_log = {
        'best_parameters': None,
        'all_runs': [],
        'search_space': {
            'gcn_depths': gcn_depths,
            'lrs': lrs,
            'convs': convs,
            'ress': ress,
            'skips': skips,
            'ends': ends,
            'layers': layers,
            'ks': ks,
            'dropouts': dropouts,
            'dilation_exs': dilation_exs,
            'node_dims': node_dims,
            'prop_alphas': prop_alphas,
            'tanh_alphas': tanh_alphas
        },
        'config': vars(args),
        'transfer_learning_used': args.use_transfer and transferred_config is not None
    }
    
    for q in range(n_iterations):
        # Random hyperparameter selection 
        gcn_depth = random.choice(gcn_depths)
        lr = random.choice(lrs)
        conv = random.choice(convs)
        res = random.choice(ress)
        skip = random.choice(skips)
        end = random.choice(ends)
        layer = random.choice(layers)
        k = random.choice(ks)
        dropout = random.choice(dropouts)
        dilation_ex = random.choice(dilation_exs)
        node_dim = random.choice(node_dims)
        prop_alpha = random.choice(prop_alphas)
        tanh_alpha = random.choice(tanh_alphas)
        
        print(f'\n Iteration {q+1}/{n_iterations} ')
        print(f'HP: gcn_depth={gcn_depth}, lr={lr}, conv={conv}, res={res}, skip={skip}, end={end}')
        print(f'     layers={layer}, k={k}, dropout={dropout}, dilation_ex={dilation_ex}')
        print(f'     node_dim={node_dim}, prop_alpha={prop_alpha}, tanh_alpha={tanh_alpha}')
        
        try:
            # Use smaller splits for 162 months
            Data = DataLoaderS(args.data, 0.7, 0.15, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)
            
            # Check if we have enough data
            if len(Data.train[0]) == 0 or len(Data.valid[0]) == 0:
                print("Not enough data for training/validation. Skipping this iteration.")
                continue
            
            # MODEL INITIALIZATION WITH TRANSFER LEARNING SUPPORT
            model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                         device, Data.adj, dropout=dropout, subgraph_size=k,
                         node_dim=node_dim, dilation_exponential=dilation_ex,
                         conv_channels=conv, residual_channels=res,
                         skip_channels=skip, end_channels=end,
                         seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                         layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
            
            # LOAD TRANSFERRED WEIGHTS IF AVAILABLE
            if args.use_transfer and transferred_config is not None:
                print("Initializing model with transferred weights...")
                try:
                    # Load the transferred model state
                    model_state = transferred_checkpoint['model_state_dict']
                    
                    # Get current model state
                    current_state = model.state_dict()
                    
                    # Filter transferred weights to only include compatible layers
                    transferred_count = 0
                    for name, param in model_state.items():
                        if name in current_state:
                            if current_state[name].shape == param.shape:
                                current_state[name] = param
                                transferred_count += 1
                    
                    # Load the filtered state dict
                    model.load_state_dict(current_state)
                    print(f"Successfully transferred {transferred_count} layers")
                    
                except Exception as e:
                    print(f"Error loading transferred weights: {e}")
                    print("Falling back to random initialization")
            
            nParams = sum([p.nelement() for p in model.parameters()])
            print(f'Number of model parameters: {nParams}')
            
            if args.L1Loss:
                criterion = nn.L1Loss(reduction='sum').to(device)
            else:
                criterion = nn.MSELoss(reduction='sum').to(device)
            evaluateL2 = nn.MSELoss(reduction='sum').to(device)
            evaluateL1 = nn.L1Loss(reduction='sum').to(device)
            
            # Use lower learning rate when using transferred model
            if args.use_transfer and transferred_config is not None:
                adjusted_lr = lr * 0.1  # Lower learning rate for fine-tuning
                print(f"Using adjusted learning rate {adjusted_lr} for fine-tuning")
            else:
                adjusted_lr = lr
            
            optim = Optim(model.parameters(), args.optim, adjusted_lr, args.clip, lr_decay=0.00001)
            
            # Training loop 
            best_val_loss = float('inf')
            es_counter = 0
            max_es_counter = 25
            
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                
                # Only validate if we have validation data
                if len(Data.valid[0]) > 0:
                    val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, 
                                                                     evaluateL2, evaluateL1, args.batch_size, False)
                else:
                    val_loss, val_rae, val_corr, val_smape = float('inf'), float('inf'), 0, float('nan')
                
                print(f'Epoch {epoch:3d} | Time: {time.time()-epoch_start_time:5.2f}s | '
                      f'Train Loss: {train_loss:5.4f} | Val RSE: {val_loss:5.4f} | '
                      f'Val RAE: {val_rae:5.4f} | Val Corr: {val_corr:5.4f} | '
                      f'Val SMAPE: {val_smape:5.4f}')
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_counter = 0
                    # Save best model for this hyperparameter set
                    torch.save(model.state_dict(), 'temp_best_model.pt')
                    current_best_metrics = (val_loss, val_rae, val_corr, val_smape)
                else:
                    es_counter += 1
                
                if es_counter >= max_es_counter:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            # Load best model for this hyperparameter set
            if os.path.exists('temp_best_model.pt'):
                model.load_state_dict(torch.load('temp_best_model.pt'))
                val_loss, val_rae, val_corr, val_smape = current_best_metrics
            else:
                val_loss, val_rae, val_corr, val_smape = float('inf'), float('inf'), 0, float('nan')
            
            # Test evaluation
            test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model,
                                                                               evaluateL2, evaluateL1, args.seq_in_len, False)
            
            print(f'Test Results - RSE: {test_acc:5.4f} | RAE: {test_rae:5.4f} | '
                  f'Corr: {test_corr:5.4f} | SMAPE: {test_smape:5.4f}')
            
            # Log this run - convert tensors to serializable types
            run_info = {
                'iteration': q,
                'hyperparameters': {
                    'gcn_depth': gcn_depth,
                    'lr': lr,
                    'adjusted_lr': adjusted_lr,
                    'conv_channels': conv,
                    'residual_channels': res,
                    'skip_channels': skip,
                    'end_channels': end,
                    'layers': layer,
                    'subgraph_size': k,
                    'dropout': dropout,
                    'dilation_exponential': dilation_ex,
                    'node_dim': node_dim,
                    'propalpha': prop_alpha,
                    'tanhalpha': tanh_alpha
                },
                'validation_metrics': {
                    'rse': float(val_loss) if not math.isinf(val_loss) else 'inf',
                    'rae': float(val_rae) if not math.isinf(val_rae) else 'inf',
                    'correlation': float(val_corr),
                    'smape': float(val_smape) if not math.isnan(val_smape) else 'nan'
                },
                'test_metrics': {
                    'rse': float(test_acc) if not math.isinf(test_acc) else 'inf',
                    'rae': float(test_rae) if not math.isinf(test_rae) else 'inf',
                    'correlation': float(test_corr),
                    'smape': float(test_smape) if not math.isnan(test_smape) else 'nan'
                },
                'parameters_count': int(nParams),
                'transfer_learning_used': args.use_transfer and transferred_config is not None
            }
            
            optimization_log['all_runs'].append(run_info)
            
            # Update best overall
            if (not math.isnan(val_corr)) and val_loss < best_rse and val_loss < float('inf'):
                best_val = val_loss + val_rae - val_corr
                best_rse = val_loss
                best_rae = val_rae
                best_corr = val_corr
                best_smape = val_smape
                best_test_rse = test_acc
                best_test_corr = test_corr
                best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, 
                          dilation_ex, node_dim, prop_alpha, tanh_alpha, layer]
                
                # Save the best model
                torch.save(model.state_dict(), args.save)
                print(f'New best model saved! Validation RSE: {val_loss:.4f}')
                
        except Exception as e:
            print(f'Error in iteration {q}: {str(e)}')
            import traceback
            traceback.print_exc()
            
            # Log failed run
            failed_run = {
                'iteration': q,
                'hyperparameters': {
                    'gcn_depth': gcn_depth,
                    'lr': lr,
                    'conv_channels': conv,
                    'residual_channels': res,
                    'skip_channels': skip,
                    'end_channels': end,
                    'layers': layer,
                    'subgraph_size': k,
                    'dropout': dropout,
                    'dilation_exponential': dilation_ex,
                    'node_dim': node_dim,
                    'propalpha': prop_alpha,
                    'tanhalpha': tanh_alpha
                },
                'error': str(e),
                'transfer_learning_used': args.use_transfer and transferred_config is not None
            }
            optimization_log['all_runs'].append(failed_run)
            continue
    
    # Final evaluation with best model 
    print('\n FINAL RESULTS ')
    if best_hp:
        print(f'Best Hyperparameters: {best_hp}')
        print(f'Best Validation - RSE: {best_rse:.4f}, RAE: {best_rae:.4f}, Correlation: {best_corr:.4f}, SMAPE: {best_smape:.4f}')
        print(f'Best Test - RSE: {best_test_rse:.4f}, Correlation: {best_test_corr:.4f}')
        
        # Load best model for final evaluation and plotting
        try:
            Data = DataLoaderS(args.data, 0.7, 0.15, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)
            model = gtnet(args.gcn_true, args.buildA_true, best_hp[0], args.num_nodes,
                         device, Data.adj, dropout=best_hp[7], subgraph_size=best_hp[6],
                         node_dim=best_hp[9], dilation_exponential=best_hp[8],
                         conv_channels=best_hp[2], residual_channels=best_hp[3],
                         skip_channels=best_hp[4], end_channels=best_hp[5],
                         seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                         layers=best_hp[12], propalpha=best_hp[10], tanhalpha=best_hp[11], layer_norm_affline=False)
            model.load_state_dict(torch.load(args.save))
            
            # Generate final plots
            print('Generating validation plots...')
            vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model,
                                                                    evaluateL2, evaluateL1, args.batch_size, True)
            
            print('Generating testing plots...')
            test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model,
                                                                               evaluateL2, evaluateL1, args.seq_in_len, True)
            
            print(' FINAL TEST RESULTS ')
            print(f'Test RSE: {test_acc:.4f} | Test RAE: {test_rae:.4f} | '
                  f'Test Correlation: {test_corr:.4f} | Test SMAPE: {test_smape:.4f}')
            
            # Save best hyperparameters
            with open('model/hp.txt', 'w') as f:
                f.write(str(best_hp))
            
            # Save optimization log with serializable data
            optimization_log['best_parameters'] = {
                'hyperparameters': best_hp,
                'validation_metrics': {
                    'rse': float(best_rse) if not math.isinf(best_rse) else 'inf',
                    'rae': float(best_rae) if not math.isinf(best_rae) else 'inf',
                    'correlation': float(best_corr),
                    'smape': float(best_smape) if not math.isnan(best_smape) else 'nan'
                },
                'test_metrics': {
                    'rse': float(best_test_rse) if not math.isinf(best_test_rse) else 'inf',
                    'correlation': float(best_test_corr)
                },
                'transfer_learning_used': args.use_transfer and transferred_config is not None
            }
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            vtest_acc, vtest_rae, vtest_corr, vtest_smape = float('nan'), float('nan'), float('nan'), float('nan')
            test_acc, test_rae, test_corr, test_smape = float('nan'), float('nan'), float('nan'), float('nan')
    else:
        print("No valid model found during optimization!")
        vtest_acc, vtest_rae, vtest_corr, vtest_smape = float('nan'), float('nan'), float('nan'), float('nan')
        test_acc, test_rae, test_corr, test_smape = float('nan'), float('nan'), float('nan'), float('nan')
    
    # Convert the entire optimization log to be JSON serializable
    serializable_log = convert_to_serializable(optimization_log)
    
    with open('model/optimization_log.json', 'w') as f:
        json.dump(serializable_log, f, indent=2)
    
    # Clean up temporary file
    if os.path.exists('temp_best_model.pt'):
        os.remove('temp_best_model.pt')
    
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape

if __name__ == "__main__":
    vacc, vrae, vcorr, vsmape, acc, rae, corr, smape = main()
    
    print('\n\n FINAL SUMMARY ')
    print("Validation Results:")
    print(f"RSE: {vacc:.4f}, RAE: {vrae:.4f}, Correlation: {vcorr:.4f}, SMAPE: {vsmape:.4f}")
    print("\nTest Results:")
    print(f"RSE: {acc:.4f}, RAE: {rae:.4f}, Correlation: {corr:.4f}, SMAPE: {smape:.4f}")