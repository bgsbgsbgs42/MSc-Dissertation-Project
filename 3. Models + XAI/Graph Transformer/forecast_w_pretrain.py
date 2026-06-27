import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import (BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout)
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.data import Data
import json
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional
import re
import pandas as pd
from datetime import datetime, timedelta

pyplot.rcParams['savefig.dpi'] = 1200

# Define the GraphTransformer model class 
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
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
            conv = GPSConv(config['channels'], GINEConv(nn_seq), heads=config.get('num_heads', 4),
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

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def consistent_name(name):
    name=name.replace('-ALL','').replace('Mentions-','').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
    
    #special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    if 'IZ' in name:
        name=name.replace('IZ', 'IS')# applicable only in our data (British English)
    if 'IOR' in name:
        name=name.replace('IOR','IOUR')#behaviour (British English)

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    
    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

def sanitize_filename(name):
    """Remove or replace invalid filename characters"""
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Remove any other non-printable characters
    name = re.sub(r'[^\x20-\x7E]', '', name)
    return name

#returns the closest curve cc to a given curve c in a list of forecasted curves, where cc is strictly larger than c
def getClosestCurveLarger(c,forecast,confidence, attack, solutions,col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==attack: #exclude irrelevant curves
            continue 
        if torch.mean(f) <= torch.mean(c):
            continue #must be larger
        if torch.mean(f)-torch.mean(c)<d:
            d=torch.mean(f)-torch.mean(c)
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf

#returns closest curve cc to given curve c in a list of forecasted curves, where cc is strictly smaller than c
def getClosestCurveSmaller(c,forecast,confidence,attack, solutions, col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==attack: #exclude irrelevant curves
            continue 
        if torch.mean(f) >= torch.mean(c):
            continue #must be smaller
        if torch.abs(torch.mean(f)-torch.mean(c))<d:
            d=torch.abs(torch.mean(f)-torch.mean(c))
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf

#negative values (due to smoothing) are changed to 0
def zero_negative_curves(data, forecast, attack, solutions, index):
    if attack in index:
        a = data[:, index[attack]]
        f= forecast[:,index[attack]]
        for i in range(a.shape[0]):
            if a[i]<0:
                a[i]=0
        for i in range(f.shape[0]):
            if f[i]<0:
                f[i]=0

    for s in solutions:
        if s in index:
            a = data[:, index[s]]
            f= forecast[:,index[s]]
            for i in range(a.shape[0]):
                if a[i]<0:
                    a[i]=0
            for i in range(f.shape[0]):
                if f[i]<0:
                    f[i]=0
    return data, forecast

def debug_data_shapes(data, forecast, attack, index):
    """Debug function to check data shapes"""
    if attack in index:
        print(f"\n=== Debug for {attack} ===")
        print(f"Full historical data shape: {data.shape}")
        print(f"Forecast data shape: {forecast.shape}")
        print(f"Historical data for {attack}: {data[:, index[attack]].shape}")
        print(f"Forecast data for {attack}: {forecast[:, index[attack]].shape}")
        print(f"Historical data sample (first 5): {data[:5, index[attack]]}")
        print(f"Historical data sample (last 5): {data[-5:, index[attack]]}")
        print(f"Forecast data sample: {forecast[:5, index[attack]]}")
           
def plot_forecast_with_dates(data, forecast, confidence, attack, solutions, index, col, dates, forecast_start_date, alarming=True):
    """Plot forecast with historical data using actual dates from data.csv"""
        
    data, forecast = zero_negative_curves(data, forecast, attack, solutions, index)
    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
        "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
        "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
        "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
        "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
        "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    
    #pyplot.style.use("seaborn-dark") 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Skip if attack not in data
    if attack not in index:
        print(f"Warning: Attack '{attack}' not found in data columns. Skipping plot.")
        pyplot.close()
        return

    # Get the full historical data for this attack
    historical_data = data[:, index[attack]]
    forecast_data = forecast[:, index[attack]]
    confidence_data = confidence[:, index[attack]]
    
    print(f"Historical data points: {len(historical_data)}")
    print(f"Forecast data points: {len(forecast_data)}")
    print(f"Available dates: {len(dates)}")
    
    # Create time indices using actual dates for historical data
    historical_time = list(range(len(historical_data)))
    
    # Generate forecast dates starting from forecast_start_date
    forecast_dates = generate_forecast_dates(forecast_start_date, len(forecast_data))
    
    all_dates = dates + forecast_dates
    all_time_indices = list(range(len(all_dates)))
    
    # Plot historical data
    a = consistent_name(attack)
    ax.plot(historical_time, historical_data, 'b-', label=f'{a} (Historical)', linewidth=2)
    
    # Plot forecast with confidence interval
    forecast_time = list(range(len(historical_data), len(historical_data) + len(forecast_data)))
    ax.plot(forecast_time, forecast_data, 'r-', label=f'{a} (Forecast)', linewidth=2)
    ax.fill_between(forecast_time, 
                forecast_data - confidence_data, 
                forecast_data + confidence_data, 
                color='red', alpha=0.3, label='95% Confidence')
    
    f_attack = torch.from_numpy(forecast_data.numpy()) if isinstance(forecast_data, torch.Tensor) else forecast_data
    counter = 1

    # Remove technologies that we are not worried about in the future
    solutions_to_plot = solutions.copy()
    if alarming:
        for s in list(solutions_to_plot):
            if s in index:
                s_forecast = forecast[:, index[s]]
                s_forecast_tensor = torch.from_numpy(s_forecast.numpy()) if isinstance(s_forecast, torch.Tensor) else s_forecast
                if torch.mean(s_forecast_tensor) >= torch.mean(f_attack):
                    solutions_to_plot.remove(s)

    # Plot the solutions
    for s in solutions_to_plot:
        if s in index:
            # Historical data for solution
            s_historical = data[:, index[s]]
            s_forecast = forecast[:, index[s]]
            s_confidence = confidence[:, index[s]]
            s_name = consistent_name(s)
            
            # Plot historical data for solution
            ax.plot(historical_time, s_historical, '--', color=colours[counter], 
                label=f'{s_name} (Historical)', linewidth=1, alpha=0.7)
            
            # Plot forecast for solution
            ax.plot(forecast_time, s_forecast, '-', color=colours[counter], 
                label=f'{s_name} (Forecast)', linewidth=1)
            
            # Plot confidence interval for solution
            ax.fill_between(forecast_time, 
                        s_forecast - s_confidence, 
                        s_forecast + s_confidence, 
                        color=colours[counter], alpha=0.3)
            
            counter += 1
            if counter >= len(colours):
                counter = 1  # Reuse colors if needed
    
    # Set x-axis labels - show every 12 months (yearly)
    year_positions = []
    year_labels = []
    
    # Dynamic x-axis labels based on forecast start year
    # Coerce dates safely and extract years
    safe_dates = []
    for d in dates:
        ds = _to_date_str(d)
        safe_dates.append(ds)

    for ds in safe_dates:
        if not ds:
            # fallback to forecast_start_date progression
            try:
                # compute year based on forecast_start_date and position
                base = pd.to_datetime(forecast_start_date)
                # index of this label (approx): compute relative month index from first safe date
                year_labels.append(str(base.year))
            except Exception:
                year_labels.append('')
        else:
            # prefer first 4 characters as year if formatted as YYYY-MM-DD
            if '-' in ds:
                year_labels.append(ds.split('-')[0])
            else:
                # fallback: take first 4 chars if they look like a year, else full string
                year_labels.append(ds[:4] if len(ds) >= 4 else ds)

    # Place ticks every 12 months if dates correspond to monthly points; otherwise evenly spaced
    try:
        # Attempt to compute tick positions as every 12th point starting at index 6 (matches previous logic)
        x_positions = [6 + 12 * i for i in range(len(year_labels))]
        ax.set_xticks(x_positions, year_labels)
    except Exception:
        # Fallback: set xticks at start, middle, end
        n = len(year_labels)
        fallback_positions = [0, n // 2, max(0, n - 1)]
        fallback_labels = [year_labels[i] if i < len(year_labels) else '' for i in fallback_positions]
        ax.set_xticks(fallback_positions)
        ax.set_xticklabels(fallback_labels)

    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 8}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True, alpha=0.3)
    pyplot.xticks(rotation=45, fontsize=10)
    
    # Create title with actual date ranges
    historical_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"
    forecast_range = f"{forecast_dates[0]} to {forecast_dates[-1]}" if forecast_dates else "N/A"
    
    pyplot.title(f"{a} Forecast\nHistorical: {historical_range} | Forecast: {forecast_range}", 
                y=1.03, fontsize=14)

    # Add vertical line to separate historical and forecast periods
    separation_point = len(historical_data)
    ax.axvline(x=separation_point - 0.5, color='gray', linestyle='--', alpha=0.7)
    ax.text(separation_point + 5, ax.get_ylim()[1] * 0.9, 'Forecast\nPeriod', 
            rotation=0, verticalalignment='top', fontsize=10, ha='center')

    fig = pyplot.gcf()
    fig.set_size_inches(14, 8)

    # Save and show the forecast
    images_dir = 'model/GraphTransformer/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    safe_filename = sanitize_filename(a)
    pyplot.savefig(images_dir + safe_filename + '_with_history.png', bbox_inches="tight", dpi=300)
    pyplot.savefig(images_dir + safe_filename + "_with_history.pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(2)
    pyplot.close()

def generate_forecast_dates(start_date, num_months):
    """Generate monthly dates starting from start_date for num_months"""
    forecast_dates = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    for i in range(num_months):
        forecast_dates.append(current_date.strftime('%Y-%m'))
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return forecast_dates

def create_graph_structure(data, num_nodes):
    """Create graph structure based on correlation between nodes"""
    # Handle NaN values in correlation calculation
    data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use correlation as adjacency measure
    try:
        correlation_matrix = np.corrcoef(data_clean.T)
        # Replace any NaN in correlation matrix with 0
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    except:
        print("Warning: Correlation calculation failed, using identity matrix")
        correlation_matrix = np.eye(num_nodes)
    
    threshold = 0.3  # Correlation threshold for edges
    adj_matrix = (correlation_matrix > threshold).astype(int)
    
    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1 and i != j:
                edge_index.append([i, j])
                edge_attr.append(1)  # Binary edge attribute
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    print(f"Graph created: {edge_index.shape[1]} edges")
    return edge_index, edge_attr

def prepare_pyg_data_single(sequence, edge_index, edge_attr, num_nodes):
    """Convert single sequence to PyG Data object for forecasting"""
    import torch_geometric.transforms as T
    
    # Create node features (current time step values for all nodes)
    x = torch.FloatTensor(sequence[-1])  # Use last time step as node features
    
    # Add positional encoding (random walk PE)
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    data = Data(
        x=x.unsqueeze(1),  # Add feature dimension
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    data = transform(data)
    return data

def create_simple_forecast(model, historical_data, edge_index, edge_attr, seq_len=12, forecast_months=36, num_samples=50):
    """Create a simple forecast using the last seq_len months of data"""
    model.eval()
    device = next(model.parameters()).device
    
    # Use the last seq_len months for forecasting
    if len(historical_data) < seq_len:
        raise ValueError(f"Not enough historical data. Need {seq_len} months, have {len(historical_data)}")
    
    input_sequence = historical_data[-seq_len:]
    
    # Prepare PyG data
    data_point = prepare_pyg_data_single(input_sequence, edge_index, edge_attr, num_nodes=historical_data.shape[1])
    
    # Generate multiple samples for uncertainty
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            data_point = data_point.to(device)
            pred = model(data_point.x, data_point.pe, data_point.edge_index, data_point.edge_attr, data_point.batch)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)  # Shape: (num_samples, forecast_months * num_nodes)
    
    # Reshape and calculate statistics
    predictions_reshaped = predictions.reshape(num_samples, forecast_months, historical_data.shape[1])
    mean_forecast = predictions_reshaped.mean(axis=0)
    std_forecast = predictions_reshaped.std(axis=0)
    
    return mean_forecast, std_forecast, predictions_reshaped

def prepare_forecast_data(historical_data, seq_len=12):
    """Prepare normalized data for forecasting"""
    # Simple normalization per column
    data_min = historical_data.min(axis=0, keepdims=True)
    data_max = historical_data.max(axis=0, keepdims=True)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # Avoid division by zero
    
    normalized_data = (historical_data - data_min) / data_range
    
    return normalized_data, data_min, data_max, data_range

def denormalize_forecast(normalized_forecast, data_min, data_max, data_range):
    """Denormalize forecast back to original scale"""
    return normalized_forecast * data_range + data_min

def debug_data_processing(historical_data, column_names, attack_name, attack_idx):
    """Debug function to understand data processing"""
    print(f"\n=== DEBUG DATA PROCESSING for {attack_name} ===")
    print(f"Raw historical data shape: {historical_data.shape}")
    print(f"Attack index: {attack_idx}")
    
    # Check raw data
    attack_data = historical_data[:, attack_idx]
    print(f"Raw data stats - Min: {attack_data.min():.6f}, Max: {attack_data.max():.6f}, Mean: {attack_data.mean():.6f}")
    print(f"Raw data sample (first 10): {attack_data[:10]}")
    print(f"Raw data sample (last 10): {attack_data[-10:]}")
    
    # Check if data has variation
    if attack_data.max() - attack_data.min() < 0.001:
        print("WARNING: Data has very little variation!")
    
    return attack_data

def save_data(data, forecast, confidence, variance, col, graph_attacks):
    """Save data only for attacks in graph.csv and their solutions"""
    # Create directories
    os.makedirs('model/GraphTransformer/forecast/data', exist_ok=True)
    
    # Get all relevant nodes (attacks + their solutions)
    relevant_nodes = set()
    for attack, solutions in graph_attacks.items():
        relevant_nodes.add(attack)
        relevant_nodes.update(solutions)
    
    # write the data and forecast only for relevant nodes
    for i in range(min(data.shape[1], len(col))):  # Use min to avoid index errors
        if i >= len(col):
            continue
            
        name = col[i]
        if name not in relevant_nodes:
            continue
            
        d = data[:,i]
        f = forecast[:,i]
        c = confidence[:,i]
        v = variance[:,i]
        
        file_dir = 'model/GraphTransformer/forecast/data/'
        safe_filename = sanitize_filename(name)
        
        try:
            with open(file_dir + safe_filename + '.txt', 'w') as ff:
                ff.write('Data: ' + str(d.tolist()) + '\n')
                ff.write('Forecast: ' + str(f.tolist()) + '\n')
                ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
                ff.write('Variance: ' + str(v.tolist()) + '\n')
        except Exception as e:
            print(f"Warning: Could not save data for {name}: {e}")

def save_gap(forecast, attack, solutions, index, col, forecast_start_date):
    """Save gap analysis with proper years based on forecast start date"""
    # Create directory
    os.makedirs('model/GraphTransformer/forecast/gap', exist_ok=True)
    
    # Skip if attack not in index
    if attack not in index:
        print(f"Warning: Attack '{attack}' not found in data. Skipping gap analysis.")
        return
    
    # Calculate year labels based on forecast start date
    start_year = int(forecast_start_date.split('-')[0])
    years = [str(start_year), str(start_year + 1), str(start_year + 2)]
    
    # write the data and forecast
    safe_filename = sanitize_filename(consistent_name(attack))
    with open('model/GraphTransformer/forecast/gap/' + safe_filename + '_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row
        writer.writerow(['Solution'] + years)
        table=[]
        a=forecast[:,index[attack]].tolist()
        a_reduced= [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]#mean of every 12 months
        for s in solutions:
            if s in index:
                row=[consistent_name(s)]
                f=forecast[:,index[s]].tolist()
                f_reduced= [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]#mean of every 12 months
                
                gap=[x - y for x, y in zip(a_reduced, f_reduced)]#calculate the gap
                row.extend(gap)#3 years gap
                table.append(row)
        if table:
            sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
            for row in sorted_table:
                writer.writerow(row)

def create_columns(file_name):
    """Given data file, returns the list of column names and dictionary of the format (column name,column index)"""
    col_name=[]
    col_index={}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name= col_name[1:]
        
        for i,c in enumerate(col_name):
            col_index[c]=i
        
        return col_name,col_index

def build_graph(file_name):
    """Builds the attacks and pertinent technologies graph"""
    # Initialise an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')
    return graph

def load_operational_model(model_path, device, num_nodes):
    """Load the operational model trained by train_operational_model.py"""
    print(f"Loading operational model from: {model_path}")
    model_data = torch.load(model_path, map_location=device)
    
    # Extract model configuration and state dict
    if 'model_state_dict' in model_data:
        state_dict = model_data['model_state_dict']
        hp = model_data.get('hyperparameters', {})
    else:
        # Assume it's just the state dict
        state_dict = model_data
        hp = {}
    
    # Ensure num_nodes matches current data
    hp['num_nodes'] = num_nodes
    
    # Set default values for required parameters
    required_params = {
        'channels': 64,
        'num_layers': 3,
        'pe_dim': 8,
        'node_dim': 1,
        'attn_type': 'multihead',
        'dropout': 0.1,
        'forecast_horizon': 36
    }
    
    for param, default in required_params.items():
        if param not in hp:
            hp[param] = default
    
    # Create model
    model = GraphTransformer(hp).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print("Operational model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting partial load...")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Partially loaded {len(pretrained_dict)} parameters")
    
    return model, hp

def _to_date_str(d):
    """Robustly convert various date-like inputs to 'YYYY-MM-DD' string or safe fallback."""
    try:
        # Strings -> return trimmed string
        if isinstance(d, str):
            return d.strip()
        # pandas Timestamp / numpy datetime / python datetime
        try:
            import pandas as _pd
            if isinstance(d, (_pd.Timestamp, datetime)):
                return _pd.to_datetime(d).strftime('%Y-%m-%d')
        except Exception:
            pass
        if isinstance(d, datetime):
            return d.strftime('%Y-%m-%d')
        # numpy datetime64
        try:
            if hasattr(d, 'dtype') and np.issubdtype(d.dtype, np.datetime64):
                return pd.to_datetime(d).strftime('%Y-%m-%d')
        except Exception:
            pass
        # Integers that look like years
        if isinstance(d, (int, np.integer)):
            if 1900 < int(d) < 3000:
                return f"{int(d)}-01-01"
            return str(int(d))
        # Floats that are year-like
        if isinstance(d, float) and not np.isnan(d):
            if abs(d - int(d)) < 1e-6 and 1900 < int(d) < 3000:
                return f"{int(d)}-01-01"
            return str(d)
        # None or NaN
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return ""
        # Fallback
        return str(d)
    except Exception:
        return str(d)

# Main execution
if __name__ == '__main__':
    import argparse
    
    # Configuration with adjustable parameters
    parser = argparse.ArgumentParser(description='Generate forecasts using operational model')
    parser.add_argument('--forecast_start_date', type=str, default='2025-01-01', 
                       help='Start date for forecasting (format: YYYY-MM-DD)')
    parser.add_argument('--forecast_months', type=int, default=36,
                       help='Number of months to forecast')
    parser.add_argument('--data_file', type=str, default='data/sm_data_g.csv',
                       help='Path to smoothed data file')
    parser.add_argument('--model_file', type=str, default='model/GraphTransformer/o_model.pt',
                       help='Path to operational model file')
    parser.add_argument('--nodes_file', type=str, default='data/sm_data_g.csv',
                       help='Path to original data file with dates')
    parser.add_argument('--graph_file', type=str, default='data/graph.csv',
                       help='Path to graph file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Forecast start date: {args.forecast_start_date}")
    print(f"Forecast period: {args.forecast_months} months")
    
    # Read the original data.csv to get historical data
    print("Loading historical data from data.csv...")
    historical_df = pd.read_csv(args.nodes_file)
    
    # Extract date column and data columns
    date_column = historical_df.iloc[:, 0]  # First column is dates
    historical_columns = historical_df.columns[1:].tolist()  # Skip date column
    historical_data_values = historical_df.iloc[:, 1:].values.astype(np.float32)  # Skip date column
    
    print(f"Historical data shape: {historical_data_values.shape}")
    print(f"Date range: {date_column.iloc[0]} to {date_column.iloc[-1]}")
    print(f"Number of columns in historical data: {len(historical_columns)}")
    
    # Create column index for historical data
    historical_index = {}
    for i, col_name in enumerate(historical_columns):
        historical_index[col_name] = i
    
    # Build the graph in the format {attack:list of pertinent technologies}
    graph = build_graph(args.graph_file)
    
    # Filter graph to only include attacks that exist in historical data
    filtered_graph = {}
    attacks_to_forecast = []
    for attack, solutions in graph.items():
        if attack in historical_index:
            # Also filter solutions to only those in historical data
            valid_solutions = [s for s in solutions if s in historical_index]
            filtered_graph[attack] = valid_solutions
            attacks_to_forecast.append(attack)
            print(f"Attack '{attack}' has {len(valid_solutions)} valid solutions")
        else:
            print(f"Warning: Attack '{attack}' not found in historical data, skipping")
    
    graph = filtered_graph
    print(f"Filtered graph has {len(graph)} attacks with valid data")
    
    # Create a subset of data containing only the attacks and solutions we care about
    relevant_columns = []
    for attack in attacks_to_forecast:
        relevant_columns.append(attack)
        relevant_columns.extend(graph[attack])
    
    relevant_columns = list(set(relevant_columns))  # Remove duplicates
    print(f"Forecasting for {len(relevant_columns)} relevant columns")
    
    # Create mapping from relevant columns to indices in the subset
    relevant_index = {}
    relevant_data = np.zeros((len(historical_data_values), len(relevant_columns)), dtype=np.float32)
    
    for i, col_name in enumerate(relevant_columns):
        relevant_index[col_name] = i
        original_idx = historical_index[col_name]
        relevant_data[:, i] = historical_data_values[:, original_idx]
    
    # Prepare data for forecasting
    print("Preparing data for forecasting...")
    normalized_data, data_min, data_max, data_range = prepare_forecast_data(relevant_data)
    
    # Create graph structure for the relevant columns only
    print("Creating graph structure for relevant columns...")
    edge_index, edge_attr = create_graph_structure(normalized_data, num_nodes=len(relevant_columns))
    
    # Load the operational model
    print("Loading operational model...")
    model, hp = load_operational_model(args.model_file, device, len(relevant_columns))
    
    # Generate forecast
    print(f"Generating {args.forecast_months}-month forecast starting from {args.forecast_start_date}...")
    mean_forecast_norm, std_forecast_norm, all_predictions = create_simple_forecast(
        model, normalized_data, edge_index, edge_attr, 
        seq_len=12, forecast_months=args.forecast_months, num_samples=50
    )
    
    print(f"Normalized forecast shape: {mean_forecast_norm.shape}")
    
    # Denormalize the forecast back to original scale
    mean_forecast = denormalize_forecast(mean_forecast_norm, data_min, data_max, data_range)
    std_forecast = denormalize_forecast(std_forecast_norm, data_min, data_max, data_range)
    
    print(f"Denormalized forecast shape: {mean_forecast.shape}")
    
    # Calculate confidence intervals (95%)
    z = 1.96
    confidence = z * std_forecast / np.sqrt(50)
    
    # Prepare data for plotting and saving
    data_tensor = torch.from_numpy(relevant_data).float()
    forecast_tensor = torch.from_numpy(mean_forecast).float()
    confidence_tensor = torch.from_numpy(confidence).float()
    
    # Save the forecast data
    print("Saving forecast data...")
    save_data(data_tensor, forecast_tensor, confidence_tensor, confidence_tensor, relevant_columns, graph)
    
    # Prepare data for plotting
    print("Preparing data for plotting...")
    
    # Use the original data for historical plotting
    historical_plot = torch.from_numpy(relevant_data).float()
    forecast_plot = torch.from_numpy(mean_forecast).float()
    
    # Apply minimal smoothing for visualization
    historical_smoothed = torch.zeros_like(historical_plot)
    forecast_smoothed = torch.zeros_like(forecast_plot)
    
    for j in range(historical_plot.shape[1]):
        # Light smoothing only
        hist_smooth = exponential_smoothing(historical_plot[:, j].numpy(), 0.03)
        fcst_smooth = exponential_smoothing(forecast_plot[:, j].numpy(), 0.03)
        historical_smoothed[:, j] = torch.tensor(hist_smooth)
        forecast_smoothed[:, j] = torch.tensor(fcst_smooth)
    
    # Normalize each column separately for better visualization, but preserve trends
    historical_normalized = torch.zeros_like(historical_smoothed)
    forecast_normalized = torch.zeros_like(forecast_smoothed)
    
    for j in range(historical_smoothed.shape[1]):
        col_data = torch.cat([historical_smoothed[:, j], forecast_smoothed[:, j]])
        col_min = col_data.min()
        col_max = col_data.max()
        col_range = col_max - col_min
        
        if col_range > 0:
            historical_normalized[:, j] = (historical_smoothed[:, j] - col_min) / col_range
            forecast_normalized[:, j] = (forecast_smoothed[:, j] - col_min) / col_range
        else:
            historical_normalized[:, j] = historical_smoothed[:, j]
            forecast_normalized[:, j] = forecast_smoothed[:, j]
    
    dates = date_column.tolist()
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Total historical months: {len(dates)}")
    
    # Plot forecasts
    print("Generating plots...")
    plotted_count = 0
    
    for attack in attacks_to_forecast:
        if attack in relevant_index:
            print(f"Plotting forecast for {attack}...")
            plot_forecast_with_dates(historical_normalized, forecast_normalized, confidence_tensor, 
                                   attack, graph[attack], relevant_index, relevant_columns, 
                                   dates, args.forecast_start_date)
            save_gap(forecast_normalized, attack, graph[attack], relevant_index, relevant_columns, args.forecast_start_date)
            plotted_count += 1
    
    print(f"Successfully generated forecasts and plots for {plotted_count} attacks")
    print(f"Forecast period: {args.forecast_start_date} to {generate_forecast_dates(args.forecast_start_date, args.forecast_months)[-1]}")