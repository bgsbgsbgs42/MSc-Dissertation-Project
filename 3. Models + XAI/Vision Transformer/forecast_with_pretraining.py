import numpy as np
import os
import torch
import torch.nn as nn
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

plt.rcParams['savefig.dpi'] = 1200

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    return name.replace('_', ' ').title()

def build_graph(file_name):
    """Build attack-technology mapping from CSV."""
    from collections import defaultdict
    graph = defaultdict(list)
    try:
        with open(file_name, 'r') as f:
            import csv
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    key_node = row[0]
                    adjacent_nodes = [node for node in row[1:] if node]
                    graph[key_node].extend(adjacent_nodes)
        print(f"Graph loaded with {len(graph)} attacks.")
    except Exception as e:
        print(f"Error loading graph: {e}")
    return graph

def create_columns(file_name):
    """Extract column names and indices from CSV file."""
    col_name = []
    col_index = {}
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)
        col_name = list(df.columns)
        
        # Remove 'Date' column if present
        if 'Date' in col_name[0] or 'date' in col_name[0].lower():
            col_name = col_name[1:]
        
        for i, c in enumerate(col_name):
            col_index[c] = i
            
    except Exception as e:
        print(f"Error reading columns: {e}")
    
    return col_name, col_index

def load_model_and_config(model_path):
    """Load the final transfer model and configuration."""
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

def generate_forecasts(model, config, data_normalized, num_forecasts=50):
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
    confidence_95 = 1.96 * std_forecast
    
    return mean_forecast, confidence_95, std_forecast

def plot_forecast(data, forecast_mean, confidence, attack, solutions, index, col_names, future_months=36):
    """Plot forecast for attack and relevant solutions."""
    # Apply smoothing and zero negative values
    data = zero_negative_curves(data)
    forecast_mean = zero_negative_curves(forecast_mean)
    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
              "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
              "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
              "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
              "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
              "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Plot historical data and forecast for attack
    counter = 0
    attack_idx = index[attack]
    
    # Historical data for attack
    hist_attack = exponential_smoothing(data[:, attack_idx])
    
    # Forecast for attack
    fcast_attack = exponential_smoothing(forecast_mean[:, attack_idx])
    conf_attack = confidence[:, attack_idx]
    
    attack_clean = consistent_name(attack)
    
    # Plot historical data
    ax.plot(range(len(hist_attack)), hist_attack, '-', color=colours[counter], 
            label=attack_clean, linewidth=2)
    
    # Plot forecast
    forecast_start = len(hist_attack)
    forecast_end = forecast_start + len(fcast_attack)
    ax.plot(range(forecast_start, forecast_end), fcast_attack, '-', 
            color=colours[counter], linewidth=2)
    
    # Plot confidence interval
    ax.fill_between(range(forecast_start, forecast_end), 
                   fcast_attack - conf_attack, 
                   fcast_attack + conf_attack, 
                   color=colours[counter], alpha=0.3)
    
    counter += 1

    # Plot solutions
    for solution in solutions:
        if solution not in index:
            continue
            
        sol_idx = index[solution]
        
        # Check if solution index is within data bounds
        if sol_idx >= data.shape[1]:
            print(f"Warning: Solution index {sol_idx} out of bounds for data shape {data.shape}")
            continue
            
        hist_sol = exponential_smoothing(data[:, sol_idx])
        fcast_sol = exponential_smoothing(forecast_mean[:, sol_idx])
        conf_sol = confidence[:, sol_idx]
        
        solution_clean = consistent_name(solution)
        
        # Plot historical data
        ax.plot(range(len(hist_sol)), hist_sol, '-', color=colours[counter % len(colours)], 
                label=solution_clean, linewidth=1.5)
        
        # Plot forecast
        ax.plot(range(forecast_start, forecast_end), fcast_sol, '--', 
                color=colours[counter % len(colours)], linewidth=1.5)
        
        # Plot confidence interval
        ax.fill_between(range(forecast_start, forecast_end), 
                       fcast_sol - conf_sol, 
                       fcast_sol + conf_sol, 
                       color=colours[counter % len(colours)], alpha=0.2)
        
        counter += 1
        
        # Highlight gaps between attack and solution forecasts
        if len(fcast_attack) == len(fcast_sol):
            gap = fcast_attack - fcast_sol
            gap_positive = gap > 0
            if np.any(gap_positive):
                x_gap = np.arange(forecast_start, forecast_end)[gap_positive]
                ax.fill_between(x_gap, 
                               fcast_attack[gap_positive], 
                               fcast_sol[gap_positive], 
                               color='orange', alpha=0.3, label='Gap' if counter == 2 else "")

    # Set x-axis labels for years
    total_months = len(hist_attack) + len(fcast_attack)
    years_needed = (total_months + 11) // 12  # Ceiling division
    years = [str(2012 + i) for i in range(years_needed)]
    year_positions = [i * 12 for i in range(len(years))]
    ax.set_xticks(year_positions[:total_months//12 + 1])
    ax.set_xticklabels(years[:total_months//12 + 1])

    ax.set_ylabel("Normalised Trend", fontsize=15)
    plt.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=13)
    plt.title(f"{attack_clean} Forecast", y=1.03, fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    # Save plot
    images_dir = 'model/ViT/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    safe_filename = attack_clean.replace('/', '_').replace('\\', '_').replace(' ', '_')
    plt.savefig(images_dir + safe_filename + '.png', bbox_inches="tight")
    plt.savefig(images_dir + safe_filename + ".pdf", bbox_inches="tight", format='pdf')
    print(f"Saved plot for {attack_clean}")
    plt.close()

def save_numerical_forecasts(forecast_mean, confidence, node_names, future_dates):
    """Save numerical forecasts to CSV."""
    os.makedirs('model/ViT/forecast/data/', exist_ok=True)
    
    # Ensure we have the right number of nodes
    num_nodes = min(forecast_mean.shape[1], len(node_names))
    
    forecast_entries = []
    for t, date in enumerate(future_dates):
        if t >= forecast_mean.shape[0]:
            break
        entry = {'date': date.strftime('%Y-%m-%d')}
        for i in range(num_nodes):
            entry[consistent_name(node_names[i])] = forecast_mean[t, i]
        forecast_entries.append(entry)
    
    df = pd.DataFrame(forecast_entries)
    df.to_csv('model/ViT/forecast/data/numerical_forecasts.csv', index=False)
    print(f"Saved numerical forecasts with shape {df.shape}")

def save_gap_analysis(forecast_mean, attack, solutions, index, future_dates):
    """Save gap analysis between attack and solutions."""
    os.makedirs('model/ViT/forecast/gap/', exist_ok=True)
    
    if attack not in index:
        return
        
    attack_idx = index[attack]
    
    # Check bounds
    if attack_idx >= forecast_mean.shape[1]:
        print(f"Warning: Attack index {attack_idx} out of bounds")
        return
        
    attack_forecast = forecast_mean[:, attack_idx]
    
    rows = []
    for solution in solutions:
        if solution not in index:
            continue
            
        sol_idx = index[solution]
        
        # Check bounds
        if sol_idx >= forecast_mean.shape[1]:
            continue
            
        sol_forecast = forecast_mean[:, sol_idx]
        gap = attack_forecast - sol_forecast
        
        for i, date in enumerate(future_dates):
            if i >= len(gap):
                break
            rows.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Attack': attack,
                'Technology': solution,
                'Gap': gap[i]
            })
    
    if rows:
        df = pd.DataFrame(rows)
        attack_clean = consistent_name(attack).replace(' ', '_').replace('/', '_').replace('\\', '_')
        df.to_csv(f'model/ViT/forecast/gap/{attack_clean}_gaps.csv', index=False)
        print(f"Saved gap analysis for {attack}")

def main():
    """Main forecasting function for final transfer model."""
    set_random_seed(123)
    
    # File paths - update for final transfer model
    model_file = 'final_transfer_model/final_model.pt'
    data_file = './data/sm_data_g.csv'  # Use the same data as training
    nodes_file = './data/sm_data_g.csv'
    graph_file = './data/graph.csv'
    
    print("=" * 80)
    print("FORECASTING WITH FINAL TRANSFER MODEL")
    print("=" * 80)
    
    try:
        # Load the final transfer model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model, config = load_model_and_config(model_file)
        
        # Prepare data for forecasting
        print("\nPreparing forecast data...")
        data, data_normalized, data_mean, data_std = prepare_forecast_data(data_file, config)
        
        # Generate forecasts
        print("\nGenerating forecasts...")
        forecasts = generate_forecasts(model, config, data_normalized, num_forecasts=50)
        
        # Calculate confidence intervals
        forecast_mean, confidence, std_forecast = calculate_confidence_intervals(forecasts)
        
        # Denormalize forecasts
        forecast_denorm = forecast_mean * data_std + data_mean
        confidence_denorm = confidence * data_std
        
        # Load column names and graph
        col_names, index = create_columns(nodes_file)
        graph = build_graph(graph_file)
        
        # Generate future dates (starting from last data point)
        last_date = datetime.strptime("31/12/2024", "%d/%m/%Y")
        future_dates = [last_date + timedelta(days=30 * i) for i in range(config['forecast_horizon'])]
        
        # Apply post-processing
        forecast_denorm = zero_negative_curves(forecast_denorm)
        data = zero_negative_curves(data)
        
        # Save numerical forecasts
        print("\nSaving numerical forecasts...")
        save_numerical_forecasts(forecast_denorm, confidence_denorm, col_names, future_dates)
        
        # Create plots for each attack in the graph
        print("\nCreating forecast plots...")
        plot_count = 0
        for attack, solutions in graph.items():
            if attack in index:
                print(f"Plotting forecast for {attack}...")
                plot_forecast(data, forecast_denorm, confidence_denorm, attack, solutions, 
                             index, col_names, future_months=config['forecast_horizon'])
                save_gap_analysis(forecast_denorm, attack, solutions, index, future_dates)
                plot_count += 1
                
                # Limit number of plots for testing (remove this for full run)
                """if plot_count >= 5:
                    print("Limited plotting to first 5 attacks for demonstration")
                    break"""
        
        print(f"\n{'='*80}")
        print("FORECASTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Created {plot_count} forecast plots")
        print(f"Numerical forecasts saved to: model/ViT/forecast/data/numerical_forecasts.csv")
        print(f"Plots saved to: model/ViT/forecast/plots/")
        print(f"Gap analysis saved to: model/ViT/forecast/gap/")
        
    except Exception as e:
        print(f"\nError during forecasting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()