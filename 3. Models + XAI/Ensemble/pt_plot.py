import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader

pyplot.rcParams['savefig.dpi'] = 1200


def exponential_smoothing(series, alpha):
    """Apply exponential smoothing to a series"""
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def consistent_name(name):
    """Format names consistently for display"""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    
    # special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

    if 'IZ' in name:
        name = name.replace('IZ', 'IS')  # applicable only in our data (British English)
    if 'IOR' in name:
        name = name.replace('IOR', 'IOUR')  # behaviour (British English)

    # e.g., University of london
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:  # e.g., "of"
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

def zero_negative_curves(data, forecast, s, index):
    """Set negative values (due to smoothing) to 0"""
    if s in index:
        a = data[:, index[s]]
        f = forecast[:, index[s]]
        for i in range(a.shape[0]):
            if a[i] < 0:
                a[i] = 0
        for i in range(f.shape[0]):
            if f[i] < 0:
                f[i] = 0
    return data, forecast

def plot_forecast(data, forecast, confidence, s, index, col):
    """Plot past data and forecast of a single pertinent technology node s"""
    data, forecast = zero_negative_curves(data, forecast, s, index)
    
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot the forecast
    if s in index:
        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)  # connect past to future
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        
        s_name = consistent_name(s)
        
        # Plot historical data
        ax.plot(range(len(d)), d, '-', color='red', label=s_name, linewidth=1)
        
        # Plot forecast
        forecast_start = len(d) - 1
        forecast_x = range(forecast_start, forecast_start + len(f))
        ax.plot(forecast_x, f, '-', color='red', linewidth=1)
        ax.fill_between(forecast_x, f - c, f + c, color='red', alpha=0.6)
        
        # Set x-axis labels for years
        x_labels = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', 
                   '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']
        x_positions = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174]
        ax.set_xticks(x_positions, x_labels)
        
        ax.set_ylabel("Trend", fontsize=15)
        pyplot.yticks(fontsize=13)
        ax.axis('tight')
        ax.grid(True)
        pyplot.xticks(rotation=90, fontsize=13)
        pyplot.title(s_name, y=1.03, fontsize=18)
        
        fig = pyplot.gcf()
        fig.set_size_inches(10, 7)
        
        # Save and show the forecast
        images_dir = 'model/Ensemble/forecast/pt_plots/'
        os.makedirs(images_dir, exist_ok=True)
        pyplot.savefig(images_dir + s_name.replace('/', '_') + '.png', bbox_inches="tight")
        pyplot.savefig(images_dir + s_name.replace('/', '_') + ".pdf", bbox_inches="tight", format='pdf')
        pyplot.show(block=False)
        pyplot.pause(5)
        pyplot.close()
    else:
        print(f"Warning: Node '{s}' not found in data columns")

def create_columns(file_name):
    """Create column names and index mapping from data file"""
    col_name = []
    col_index = {}
    
    try:
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            col_name = [c for c in next(reader)]
            if 'Date' in col_name[0]:
                col_name = col_name[1:]
            
            for i, c in enumerate(col_name):
                col_index[c] = i
    except Exception as e:
        print(f"Warning: Could not read column names from {file_name}: {e}")
        print("Using generated column names")
    
    return col_name, col_index

def build_graph(file_name):
    """Build the attacks and pertinent technologies graph"""
    graph = defaultdict(list)
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                key_node = row[0]
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
    
    print(f'Graph loaded with {len(graph)} attacks...')
    return graph

def load_ensemble_model(model_path, config_path, device):
    """Load the trained ensemble model"""
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None, None, None, None
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Loaded checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract hyperparameters and model state
        if 'hyperparameters' in checkpoint:
            config = checkpoint['hyperparameters']
            model_state_dict = checkpoint['model_state_dict']
            scale = checkpoint.get('scale', None)
            feature_names = checkpoint.get('feature_names', None)
            print("Loaded model with hyperparameters from checkpoint")
        else:
            # Fallback for older format
            print("Using fallback model loading")
            config = {}
            model_state_dict = checkpoint
            scale = None
            feature_names = None
        
        # If config_path is provided, load additional config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            print("Loaded additional config from file")
        
        # Ensure required config parameters
        required_params = ['sequence_length', 'forecast_horizon', 'num_nodes']
        for param in required_params:
            if param not in config:
                if param == 'num_nodes':
                    # Try to infer from model state dict
                    for key in model_state_dict.keys():
                        if 'graph_net' in key and 'weight' in key:
                            config['num_nodes'] = model_state_dict[key].shape[0]
                            print(f"Inferred num_nodes: {config['num_nodes']}")
                            break
                    if 'num_nodes' not in config:
                        config['num_nodes'] = 1292  # default
                else:
                    # Use defaults
                    default_config = {
                        'sequence_length': 12,
                        'forecast_horizon': 36,
                        'num_nodes': 1292,
                        'vit_patch_size': 4,
                        'vit_embed_dim': 128,
                        'vit_num_heads': 8,
                        'vit_hidden_dim': 256,
                        'vit_num_layers': 3,
                        'graph_hidden_dim': 128,
                        'graph_num_layers': 2,
                        'fusion_dim': 256,
                        'dropout': 0.1,
                        'mc_dropout': 0.2
                    }
                    config[param] = default_config[param]
                    print(f"Using default {param}: {config[param]}")
        
        # Build model - try different import approaches
        try:
            from hyperparameter_optimization_ensemble_pretraining import SpatioTemporalEnsemble
        except ImportError:
            print("Warning: Could not import SpatioTemporalEnsemble from module")
            # Create a fallback model definition
            class SpatioTemporalEnsemble(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    # Simple fallback model
                    self.forecast_net = nn.Linear(
                        config['sequence_length'] * config['num_nodes'], 
                        config['forecast_horizon'] * config['num_nodes']
                    )
                
                def forward(self, x, mc_dropout=False):
                    batch_size, seq_len, num_nodes = x.shape
                    x = x.reshape(batch_size, -1)
                    output = self.forecast_net(x)
                    output = output.reshape(batch_size, self.config['forecast_horizon'], num_nodes)
                    return output
        
        model = SpatioTemporalEnsemble(config).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print("Ensemble model loaded successfully")
        print(f"Model configured for {config['forecast_horizon']}-month forecast")
        
        return model, config, scale, feature_names
        
    except Exception as e:
        print(f"Error loading ensemble model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def prepare_forecast_data(data_path, sequence_length, device, scale=None):
    """Prepare data for forecasting using the same normalization as training"""
    # Load data with proper handling of string values
    if data_path.endswith('.csv'):
        # Use pandas to read and convert to numeric, handling errors
        data_df = pd.read_csv(data_path, header=None)
        # Convert all columns to numeric, coercing errors to NaN
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        # Fill NaN values with 0 or forward fill based on your data preference
        data_df = data_df.fillna(0)
        data = data_df.values
    else:
        # For non-CSV files, use delim_whitespace with numeric conversion
        data_df = pd.read_csv(data_path, header=None, delim_whitespace=True)
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        data_df = data_df.fillna(0)
        data = data_df.values
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Ensure data is float type
    data = data.astype(np.float32)
    
    # Normalize data using the same scaling as training
    if scale is not None:
        # Use the scale from training
        data_normalized = data / scale
        print("Used training scale for normalization")
    else:
        # Fallback: normalize using data statistics
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0) + 1e-8
        data_normalized = (data - data_mean) / data_std
        print("Used data statistics for normalization")
    
    # Use the last sequence_length points for forecasting
    forecast_input = data_normalized[-sequence_length:]
    
    # Convert to tensor and add batch dimension
    forecast_input = torch.FloatTensor(forecast_input).unsqueeze(0).to(device)
    
    return forecast_input, data

def ensemble_forecast(model, input_data, num_runs=10):
    """Generate forecast using ensemble model with uncertainty estimation"""
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Use MC dropout for uncertainty estimation
            output = model(input_data, mc_dropout=True)
            outputs.append(output)
    
    # Stack outputs and compute statistics
    outputs = torch.stack(outputs)
    forecast_mean = torch.mean(outputs, dim=0)
    forecast_std = torch.std(outputs, dim=0)
    
    # 95% confidence interval
    z = 1.96
    confidence = z * forecast_std / torch.sqrt(torch.tensor(num_runs, dtype=torch.float32))
    
    return forecast_mean.squeeze(0), confidence.squeeze(0), outputs

def main():
    """Main function for plotting ensemble model forecasts"""
    data_file = 'Dissertation/Ensemble Variant/data/sm_data_g.csv'
    model_file = 'Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt'
    config_file = 'Dissertation/Ensemble Variant/model/Ensemble/hp.txt'
    nodes_file = 'Dissertation/Ensemble Variant/data/sm_data_g.csv'
    graph_file = 'Dissertation/Ensemble Variant/data/graph.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load column names and build graph
    col, index = create_columns(nodes_file)
    print(f"Loaded {len(col)} column names from data file")
    
    graph = build_graph(graph_file)
    print(f"Built graph with {len(graph)} attacks")
    
    # Load ensemble model
    print("Loading ensemble model...")
    model, config, scale, feature_names = load_ensemble_model(model_file, config_file, device)
    
    if model is None:
        print("CRITICAL ERROR: Failed to load ensemble model. Cannot proceed with forecasting.")
        print("Please check:")
        print(f"1. Model file exists: {os.path.exists(model_file)}")
        print(f"2. Model file path: {model_file}")
        return
    
    print("Preparing forecast data...")
    forecast_input, full_data = prepare_forecast_data(
        data_file, config['sequence_length'], device, scale
    )
    
    # Generate forecast for 2025-2027 (36 months)
    print("Generating forecast for 2025-2027 (36 months)...")
    forecast, confidence, all_outputs = ensemble_forecast(model, forecast_input, num_runs=10)
    
    # Denormalize forecast
    if scale is not None:
        # Use the training scale for denormalization
        scale_tensor = torch.FloatTensor(scale).to(device)
        forecast = forecast * scale_tensor
        confidence = confidence * scale_tensor
    else:
        # Fallback: use data statistics
        data_mean = full_data.mean(axis=0)
        data_std = full_data.std(axis=0) + 1e-8
        forecast = forecast.cpu() * data_std + data_mean
        confidence = confidence.cpu() * data_std
    
    full_data_tensor = torch.FloatTensor(full_data)
    
    print(f"Historical data shape: {full_data_tensor.shape}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    # Combine historical and forecast data
    all_combined = torch.cat((full_data_tensor, forecast), dim=0)
    
    # Global normalization for consistent scaling
    max_val = torch.max(all_combined)
    all_normalized = all_combined / max_val
    confidence_normalized = confidence / max_val
    
    # Apply exponential smoothing
    smoothed_data = torch.zeros_like(all_normalized)
    for i in range(all_normalized.shape[1]):
        smoothed_col = exponential_smoothing(all_normalized[:, i].tolist(), 0.1)
        smoothed_data[:, i] = torch.FloatTensor(smoothed_col)
    
    smoothed_confidence = torch.zeros_like(confidence_normalized)
    for i in range(confidence_normalized.shape[1]):
        smoothed_conf = exponential_smoothing(confidence_normalized[:, i].tolist(), 0.1)
        smoothed_confidence[:, i] = torch.FloatTensor(smoothed_conf)
    
    # Split back into historical and forecast periods
    historical_smoothed = smoothed_data[:-36]
    forecast_smoothed = smoothed_data[-36:]
    confidence_smoothed = smoothed_confidence
    
    # Plot all forecasted pertinent technologies
    print("Generating individual technology plots...")
    done = []
    plotted_count = 0
    
    for attack, solutions in graph.items():
        for s in solutions:
            if s not in done and s in index:
                plot_forecast(historical_smoothed, forecast_smoothed, confidence_smoothed, 
                             s, index, col)
                done.append(s)
                plotted_count += 1
            elif s not in index:
                print(f"Skipping '{s}' - not found in data columns")
    
    print(f"Plotting completed successfully! Generated {plotted_count} individual technology plots.")
    print(f"Results saved in model/Ensemble/forecast/pt_plots/")
    print(f"Forecast period: 36 months (2025-2027)")

if __name__ == "__main__":
    main()