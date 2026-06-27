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
from matplotlib import pyplot as plt
import random
import pandas as pd
from datetime import datetime, timedelta
import math

plt.rcParams['savefig.dpi'] = 1200

# Import the exact model architecture from forecast_with_pretraining.py
class PatchEmbedding(nn.Module):
    """Convert time series into patches with proper size handling"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Ensure sequence length is divisible by patch size
        if seq_len % patch_size != 0:
            # Find the largest patch size that divides sequence length
            divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
            if divisors:
                patch_size = max(divisors)  # Use largest divisor for stability
                print(f"Adjusted patch_size to: {patch_size} (divisor of {seq_len})")
            else:
                patch_size = 1  # Fallback
                print(f"Using fallback patch_size: 1")
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        print(f"Final: seq_len={seq_len}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Proper 1D convolution for time series
        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Initialize with smaller values for stability
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Position embeddings match actual num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes)
        batch_size = x.shape[0]
        
        # Reshape for conv1d: (batch_size, num_nodes, seq_len)
        x = x.transpose(1, 2)
        
        # Apply 1D convolution: (batch_size, embed_dim, num_patches)
        x = self.projection(x)
        
        # Transpose back: (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings - sizes should match now
        x = x + self.position_embeddings
        
        return x

class VisionTransformerForTimeSeries(nn.Module):
    """Vision Transformer adapted for time series forecasting"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self._validate_config(config)
        
        # Calculate actual patch size and num_patches before creating patch_embed
        actual_patch_size = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
        actual_num_patches = config['sequence_length'] // actual_patch_size
        
        print(f"Model config: sequence_length={config['sequence_length']}, patch_size={actual_patch_size}, num_patches={actual_num_patches}, forecast_horizon={config['forecast_horizon']}")
        
        self.patch_embed = PatchEmbedding(
            seq_len=config['sequence_length'],
            patch_size=actual_patch_size,
            in_channels=config['num_nodes'],
            embed_dim=config['embed_dim']
        )
        
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
        
        # Initialize forecast head with smaller weights
        self.forecast_head = nn.Sequential(
            nn.Linear(config['embed_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['forecast_horizon'] * config['num_nodes'])
        )
        
        # Initialize weights properly
        self._init_weights()
        
        self.mc_dropout = nn.Dropout(config['mc_dropout'])
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
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
        
        # Ensure compatibility before creating layers
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
            return 1  # Fallback
        
        # Use the largest divisor for stability (not the closest)
        return max(divisors)
    
    def _get_compatible_heads(self, embed_dim, desired_heads):
        """Find compatible number of heads that divides embed_dim"""
        divisors = []
        for i in range(1, embed_dim + 1):
            if embed_dim % i == 0:
                divisors.append(i)
        
        if not divisors:
            return 1  # Fallback
        
        # Prefer smaller number of heads for stability
        compatible_heads = [h for h in divisors if h <= min(desired_heads, 16)]
        if compatible_heads:
            return max(compatible_heads)
        else:
            return min(divisors)
        
    def forward(self, x, mc_dropout=True):
        # x shape: (batch_size, seq_len, num_nodes)
        x = self.patch_embed(x)
        
        x = self.transformer(x)
        
        cls_token = x[:, 0]
        
        if mc_dropout:
            cls_token = self.mc_dropout(cls_token)
            
        forecast = self.forecast_head(cls_token)
        forecast = forecast.view(-1, self.config['forecast_horizon'], self.config['num_nodes'])
        
        return forecast

# FORECASTING FUNCTIONS
def exponential_smoothing(series, alpha=0.3):
    """Apply exponential smoothing to time series."""
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

def zero_negative_curves(data):
    """Clip negative values to zero."""
    if isinstance(data, torch.Tensor):
        data = torch.clamp(data, min=0)
    else:
        data = np.maximum(data, 0)
    return data

def consistent_name(name):
    """Clean and format names for consistency."""
    if not isinstance(name, str):
        return str(name)
    
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    
    # Special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

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

def load_model_and_config(model_path):
    """Load the Vision Transformer model and configuration."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the saved model data
        model_data = torch.load(model_path, map_location='cpu')
        
        # Extract configuration and state dict
        config = model_data['hyperparameters']
        state_dict = model_data['model_state_dict']
        
        print(f"Loaded model configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create model with the loaded configuration
        model = VisionTransformerForTimeSeries(config)
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        print("Model loaded successfully!")
        return model, config
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_forecast_data(data_path, config):
    """Prepare data for forecasting."""
    print(f"Loading data from: {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Remove date column if present
    if 'Date' in data.columns or 'date' in data.columns.str.lower():
        data = data.iloc[:, 1:]  # Remove first column (dates)
    
    data = data.values
    print(f"Data shape: {data.shape}")
    
    # Update config with actual data dimensions
    actual_num_nodes = data.shape[1]
    if 'num_nodes' in config and config['num_nodes'] != actual_num_nodes:
        print(f"Warning: Config expects {config['num_nodes']} nodes, but data has {actual_num_nodes}")
        print(f"Updating config to match data...")
        config['num_nodes'] = actual_num_nodes
    
    # Normalize data
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_std[data_std == 0] = 1.0  # Avoid division by zero
    
    data_normalized = (data - data_mean) / data_std
    
    print(f"Data normalized. Mean shape: {data_mean.shape}, Std shape: {data_std.shape}")
    
    return data, data_normalized, data_mean, data_std

def generate_vit_forecasts(model, config, data_normalized, num_forecasts=10):
    """Generate multiple forecasts with MC dropout for uncertainty estimation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    model.eval()  # Set to eval mode but keep dropout active for MC
    
    # Use the last sequence from data
    sequence_length = config['sequence_length']
    
    # Ensure we have enough data
    if len(data_normalized) < sequence_length:
        raise ValueError(f"Not enough data. Have {len(data_normalized)} samples, need {sequence_length}")
    
    last_sequence = data_normalized[-sequence_length:]
    last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    
    print(f"Input sequence shape: {last_sequence.shape}")
    print(f"Generating {num_forecasts} forecasts with MC dropout...")
    
    # Generate multiple forecasts with MC dropout
    forecasts = []
    with torch.no_grad():
        for i in range(num_forecasts):
            forecast = model(last_sequence, mc_dropout=True)
            forecasts.append(forecast.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_forecasts} forecasts")
    
    forecasts = np.array(forecasts)  # (num_forecasts, 1, horizon, num_nodes)
    forecasts = forecasts.squeeze(1)  # Remove batch dimension -> (num_forecasts, horizon, num_nodes)
    
    print(f"Final forecasts shape: {forecasts.shape}")
    return forecasts

def calculate_confidence_intervals(forecasts):
    """Calculate mean and confidence intervals from multiple forecasts."""
    mean_forecast = np.mean(forecasts, axis=0)
    std_forecast = np.std(forecasts, axis=0)
    confidence_95 = 1.96 * std_forecast / np.sqrt(len(forecasts))
    
    return mean_forecast, confidence_95, std_forecast

#given data file, returns the list of column names and dictionary of the format (column name,column index)
def create_columns(file_name):
    col_name=[]
    col_index={}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name = col_name[1:]
        
        for i, c in enumerate(col_name):
            col_index[c] = i
        
        return col_name, col_index

#builds the attacks and pertinent technologies graph
def build_graph(file_name):
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
            adjacent_nodes = [node for node in row[1:] if node]  # does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with', len(graph), 'attacks...')
    return graph

#plots past data and forecast of a single pertinent technology node s
def plot_forecast(data, forecast, confidence, s, index, col):
    """Plot forecast for a single technology/attack node."""
    # Apply smoothing and zero negative values
    data = exponential_smoothing(data, 0.1)
    forecast = exponential_smoothing(forecast, 0.1)
    confidence = exponential_smoothing(confidence, 0.1)
    
    data = zero_negative_curves(data)
    forecast = zero_negative_curves(forecast)
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot the forecast
    if s not in index:
        print(f"Warning: {s} not found in index")
        plt.close()
        return
        
    node_idx = index[s]
    
    # Ensure indices are within bounds
    if node_idx >= data.shape[1]:
        print(f"Warning: Index {node_idx} out of bounds for data shape {data.shape}")
        plt.close()
        return
    
    # Historical data for the node
    hist_data = data[:, node_idx]
    
    # Forecast for the node
    fcast_data = forecast[:, node_idx]
    conf_data = confidence[:, node_idx]
    
    s_clean = consistent_name(s)
    
    # Connect the past to future in the plot
    d = np.concatenate([hist_data, fcast_data[:1]])  # Connect past to future
    
    # Plot historical data
    ax.plot(range(len(hist_data)), hist_data, '-', color='red', label=s_clean, linewidth=1)
    
    # Plot forecast
    forecast_start = len(hist_data)
    forecast_end = forecast_start + len(fcast_data)
    ax.plot(range(forecast_start, forecast_end), fcast_data, '-', color='red', linewidth=1)
    
    # Plot confidence interval
    ax.fill_between(range(forecast_start, forecast_end), 
                   fcast_data - conf_data, 
                   fcast_data + conf_data, 
                   color='red', alpha=0.6)
    
    # Set x-axis labels for years
    total_months = len(hist_data) + len(fcast_data)
    years_needed = (total_months + 11) // 12  # Ceiling division
    years = [str(2012 + i) for i in range(years_needed)]
    year_positions = [i * 12 for i in range(len(years))]
    ax.set_xticks(year_positions[:total_months//12 + 1])
    ax.set_xticklabels(years[:total_months//12 + 1])

    ax.set_ylabel("Trend", fontsize=15)
    plt.yticks(fontsize=13)
    ax.axis('tight')
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=13)
    plt.title(s_clean, y=1.03, fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    # Save and show the forecast
    images_dir = 'model/ViT/forecast/pt_plots/'
    os.makedirs(images_dir, exist_ok=True)
    
    safe_filename = s_clean.replace('/', '_').replace('\\', '_').replace(' ', '_')
    plt.savefig(os.path.join(images_dir, safe_filename + '.png'), bbox_inches="tight")
    plt.savefig(os.path.join(images_dir, safe_filename + ".pdf"), bbox_inches="tight", format='pdf')
    
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def main():
    """Main function for Vision Transformer pt_plotting."""
    print("=" * 80)
    print("VISION TRANSFORMER PT_PLOTTING")
    print("=" * 80)
    
    # File paths - using the final transfer model
    model_file = 'Dissertation/Vision Transformer/final_transfer_model/final_model.pt'
    data_file = 'Dissertation/Vision Transformer/data/sm_data_g.csv'
    nodes_file = 'Dissertation/Vision Transformer/data/sm_data_g.csv'
    graph_file = 'Dissertation/Vision Transformer/data/graph.csv'
    
    try:
        # Load the Vision Transformer model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model, config = load_model_and_config(model_file)
        
        # Prepare data for forecasting
        print("\nPreparing forecast data...")
        data, data_normalized, data_mean, data_std = prepare_forecast_data(data_file, config)
        
        # Generate forecasts using Vision Transformer
        print("\nGenerating forecasts with Vision Transformer...")
        forecasts = generate_vit_forecasts(model, config, data_normalized, num_forecasts=10)
        
        # Calculate confidence intervals
        forecast_mean, confidence, std_forecast = calculate_confidence_intervals(forecasts)
        
        # Denormalize forecasts
        forecast_denorm = forecast_mean * data_std + data_mean
        confidence_denorm = confidence * data_std
        
        # Load column names and graph
        col_names, index = create_columns(nodes_file)
        graph = build_graph(graph_file)
        
        # Apply post-processing
        forecast_denorm = zero_negative_curves(forecast_denorm)
        data = zero_negative_curves(data)
        
        # Combine historical data with forecast for plotting
        all_data = np.vstack([data, forecast_denorm])
        
        # Normalize for consistent plotting (global normalization)
        incident_max = -np.inf
        mention_max = -np.inf
        
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if j >= len(col_names):
                    continue
                    
                col_name = col_names[j]
                if 'WAR' in col_name or 'Holiday' in col_name or (16 <= j < 32):
                    continue
                    
                if 'Mention' in col_name:
                    if all_data[i, j] > mention_max:
                        mention_max = all_data[i, j]
                else:
                    if all_data[i, j] > incident_max:
                        incident_max = all_data[i, j]
        
        # Apply normalization
        all_norm = np.zeros_like(all_data)
        confidence_norm = np.zeros_like(confidence_denorm)
        
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if j >= len(col_names):
                    continue
                    
                col_name = col_names[j]
                if 'Mention' in col_name and mention_max > 0:
                    all_norm[i, j] = all_data[i, j] / mention_max
                elif incident_max > 0:
                    all_norm[i, j] = all_data[i, j] / incident_max
                
                # Normalize confidence intervals
                if i >= all_data.shape[0] - config['forecast_horizon']:
                    u = i - (all_data.shape[0] - config['forecast_horizon'])
                    if all_data[i, j] != 0:
                        scale_factor = all_norm[i, j] / all_data[i, j]
                        confidence_norm[u, j] = confidence_denorm[u, j] * scale_factor
        
        # Split back into historical and forecast parts
        hist_norm = all_norm[:-config['forecast_horizon'], :]
        forecast_norm = all_norm[-config['forecast_horizon']:, :]
        
        print(f"\nPlotting forecasts for pertinent technologies...")
        
        # Track plotted technologies
        done = []
        
        # Plot forecasts for all pertinent technologies
        for attack, solutions in graph.items():
            for solution in solutions:
                if solution not in done and solution in index:
                    done.append(solution)
                    print(f"Plotting forecast for {solution}...")
                    plot_forecast(hist_norm, forecast_norm, confidence_norm, solution, index, col_names)
        
        print(f"\n{'='*80}")
        print("PT_PLOTTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Created {len(done)} individual forecast plots")
        print(f"Plots saved to: model/ViT/forecast/pt_plots/")
        
    except Exception as e:
        print(f"\nError during pt_plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()