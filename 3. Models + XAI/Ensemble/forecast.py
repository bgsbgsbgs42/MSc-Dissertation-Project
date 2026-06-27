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

# Import ensemble model components
try:
    from hyperparameter_optimization_ensemble_pretraining import (
        SpatioTemporalEnsemble, CyberThreatDataset, 
        EnsembleCyberThreatForecaster, NumpyEncoder
    )
except ImportError:
    # Fallback definitions if imports fail
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

    class CyberThreatDataset(Dataset):
        def __init__(self, data, sequence_length=12, forecast_horizon=36):
            if isinstance(data, (pd.DataFrame, pd.Series)):
                self.data = data.values
            else:
                self.data = data
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon
            self.num_nodes = self.data.shape[1]
            
        def __len__(self):
            total_length = len(self.data) - self.sequence_length - self.forecast_horizon + 1
            return max(0, total_length)
        
        def __getitem__(self, idx):
            x = self.data[idx:idx + self.sequence_length]
            y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
            return torch.FloatTensor(x), torch.FloatTensor(y)

def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def consistent_name(name):
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

def getClosestCurveLarger(c, forecast, confidence, attack, solutions, index):
    """Find closest curve that is larger than c, using only specified solutions"""
    d = 999999999
    cc = None
    cc_conf = None
    
    for solution in solutions:
        if solution in index:
            j = index[solution]
            f = forecast[:, j]
            f_conf = confidence[:, j]
            
            if torch.mean(f) <= torch.mean(c):
                continue  # must be larger
            if torch.mean(f) - torch.mean(c) < d:
                d = torch.mean(f) - torch.mean(c)
                cc = f.clone()
                cc_conf = f_conf.clone()
                
    return cc, cc_conf

def getClosestCurveSmaller(c, forecast, confidence, attack, solutions, index):
    """Find closest curve that is smaller than c, using only specified solutions"""
    d = 999999999
    cc = None
    cc_conf = None
    
    for solution in solutions:
        if solution in index:
            j = index[solution]
            f = forecast[:, j]
            f_conf = confidence[:, j]
            
            if torch.mean(f) >= torch.mean(c):
                continue  # must be smaller
            if torch.abs(torch.mean(f) - torch.mean(c)) < d:
                d = torch.abs(torch.mean(f) - torch.mean(c))
                cc = f.clone()
                cc_conf = f_conf.clone()
                
    return cc, cc_conf

def zero_negative_curves(data, forecast, attack, solutions, index):
    if attack in index:
        a = data[:, index[attack]]
        f = forecast[:, index[attack]]
        for i in range(a.shape[0]):
            if a[i] < 0:
                a[i] = 0
        for i in range(f.shape[0]):
            if f[i] < 0:
                f[i] = 0

    for s in solutions:
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

def plot_forecast_ensemble(data, forecast, confidence, attack, solutions, index, alarming=True):
    data, forecast = zero_negative_curves(data, forecast, attack, solutions, index)
    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
          "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
          "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
          "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
          "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
          "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]
 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Plot historical data from July 2011 to Dec 2024
    historical_start = 6  # July 2011 (0-based index: 0=Jan 2011, 6=Jul 2011)
    historical_end = data.shape[0]  # Use all available historical data
    
    # Plot the historical data and forecast of attack
    counter = 0
    
    if attack in index:
        historical_attack = data[historical_start:historical_end, index[attack]]
        forecast_attack = forecast[:, index[attack]]
        confidence_attack = confidence[:, index[attack]]
        
        a = consistent_name(attack)
        
        # Plot historical data
        ax.plot(range(len(historical_attack)), historical_attack, '-', color=colours[counter], label=a, linewidth=2)
        
        # Plot forecast (36 months for 2025-2027)
        forecast_start = len(historical_attack)
        forecast_x = range(forecast_start, forecast_start + len(forecast_attack))
        ax.plot(forecast_x, forecast_attack, '-', color=colours[counter], linewidth=2)
        ax.fill_between(forecast_x, forecast_attack - confidence_attack, forecast_attack + confidence_attack, 
                       color=colours[counter], alpha=0.6)
        
        f_attack = forecast_attack.clone()
        counter += 1
    else:
        print(f"Warning: Attack '{attack}' not found in data columns")
        return

    # Remove technologies that we are not worried about in the future
    valid_solutions = []
    for s in solutions:
        if s in index:
            f = forecast[:, index[s]]
            if not alarming or (alarming and torch.mean(f) < torch.mean(f_attack)):
                valid_solutions.append(s)
        else:
            print(f"Warning: Solution '{s}' not found in data columns")

    # Plot the forecast of the solutions
    for s in valid_solutions:
        historical_solution = data[historical_start:historical_end, index[s]]
        forecast_solution = forecast[:, index[s]]
        confidence_solution = confidence[:, index[s]]
        s_name = consistent_name(s)
        
        # Plot historical data
        ax.plot(range(len(historical_solution)), historical_solution, '-', color=colours[counter], label=s_name, linewidth=1)
        
        # Plot forecast
        ax.plot(forecast_x, forecast_solution, '-', color=colours[counter], linewidth=1)
        ax.fill_between(forecast_x, forecast_solution - confidence_solution, forecast_solution + confidence_solution, 
                       color=colours[counter], alpha=0.6)
        
        if torch.mean(f_attack) > torch.mean(forecast_solution):
            cc, cc_conf = getClosestCurveLarger(forecast_solution, forecast, confidence, attack, valid_solutions, index)
            if cc is not None:
                ax.fill_between(forecast_x, cc - cc_conf, forecast_solution + confidence_solution, 
                               color=colours[counter], alpha=0.3)
        else:
            cc, cc_conf = getClosestCurveSmaller(forecast_solution, forecast, confidence, attack, valid_solutions, index)
            if cc is not None:
                ax.fill_between(forecast_x, cc + cc_conf, forecast_solution - confidence_solution, 
                               color=colours[counter], alpha=0.3)

        counter += 1  
    
    # Set x-axis labels for years
    total_months = len(historical_attack) + len(forecast_attack)
    start_year = 2011
    start_month = 7  # July
    years = []
    year_positions = []
    
    current_year = start_year
    current_month = start_month
    position = 0
    
    while position <= total_months:
        years.append(str(current_year))
        year_positions.append(position)
        current_year += 1
        position += 12
    
    ax.set_xticks(year_positions, years)
    
    # Highlight forecast period
    ax.axvspan(forecast_start, forecast_start + len(forecast_attack) - 1, color="skyblue", alpha=0.3, label="Forecast Period")
    
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(f"{a} - Historical & Forecast", y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(12, 7) 

    # Save and show the forecast
    images_dir = 'model/Ensemble/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    pyplot.savefig(images_dir + a.replace('/', '_') + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + a.replace('/', '_') + ".pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(2)
    pyplot.close()

def save_data_ensemble(data, forecast, confidence, graph_columns, index):
    """Save the data and forecast only for columns present in graph.csv"""
    os.makedirs('model/Ensemble/forecast/data/', exist_ok=True)
    
    # Only save data for columns that are in the graph
    saved_count = 0
    for column_name in graph_columns:
        if column_name in index:
            i = index[column_name]
            d = data[:, i]
            f = forecast[:, i]
            c = confidence[:, i]
            
            file_dir = 'model/Ensemble/forecast/data/'
            with open(file_dir + column_name.replace('/', '_') + '.txt', 'w') as ff:
                ff.write('Data: ' + str(d.tolist()) + '\n')
                ff.write('Forecast: ' + str(f.tolist()) + '\n')
                ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            saved_count += 1
    
    print(f"Saved forecast data for {saved_count} graph columns")

def save_gap_ensemble(forecast, attack, solutions, index):
    """Save the forecasted trend gaps for ensemble model"""
    os.makedirs('model/Ensemble/forecast/gap/', exist_ok=True)
    
    if attack not in index:
        print(f"Warning: Cannot save gap for '{attack}' - not found in data")
        return
        
    with open('model/Ensemble/forecast/gap/' + consistent_name(attack).replace('/', '_') + '_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Solution', '2025', '2026', '2027'])
        table = []
        a = forecast[:, index[attack]].tolist()
        a_reduced = [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]  # mean of every 12 months
        
        for s in solutions:
            if s in index:
                row = [consistent_name(s)]
                f = forecast[:, index[s]].tolist()
                f_reduced = [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]  # mean of every 12 months
                
                gap = [x - y for x, y in zip(a_reduced, f_reduced)]  # calculate the gap
                row.extend(gap)  # 3 years gap (2025-2027)
                table.append(row)
        
        if table:
            sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
            for row in sorted_table:
                writer.writerow(row)
            print(f"Saved gap analysis for {attack} with {len(table)} solutions")
        else:
            print(f"No valid solutions found for gap analysis of {attack}")

def create_columns(file_name):
    """Create column names and index mapping"""
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

def build_graph_and_get_columns(file_name):
    """Build graph and return all unique column names from first row of graph.csv"""
    graph = defaultdict(list)
    all_columns = set()

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                key_node = row[0]
                all_columns.add(key_node)
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
                # Add all solutions to the column set
                for node in adjacent_nodes:
                    all_columns.add(node)
    
    print(f'Graph loaded with {len(graph)} attacks...')
    print(f'Total unique columns in graph: {len(all_columns)}')
    return graph, list(all_columns)

def load_ensemble_model(model_path, config_path, device):
    """Load the trained ensemble model"""
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None, None, None, None
        
        # Load the checkpoint saved by train_final_ensemble.py
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
            print("Warning: Could not import SpatioTemporalEnsemble")
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
        
        print("Ensemble model loaded successfully from train_final_ensemble.py")
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
    print(f"Sample data: {data[0, :5]}")  # Print first 5 values of first row for debugging
    
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
    data_file = 'Dissertation/Ensemble Variant/data/sm_data_g.csv'
    model_file = 'Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt'  
    config_file = 'Dissertation/Ensemble Variant/model/Ensemble/hp.txt'
    nodes_file = 'Dissertation/Ensemble Variant/data/sm_data_g.csv'
    graph_file = 'Dissertation/Ensemble Variant/data/graph.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load column names and build graph with all unique columns
    col, index = create_columns(nodes_file)
    print(f"Loaded {len(col)} column names from data file")
    
    graph, graph_columns = build_graph_and_get_columns(graph_file)
    print(f"Found {len(graph_columns)} unique columns in graph file")
    
    # Check which graph columns exist in our data
    available_columns = [col for col in graph_columns if col in index]
    print(f"Available columns in data: {len(available_columns)}")
    
    if len(available_columns) == 0:
        print("Warning: No graph columns found in data file!")
        print("First 10 graph columns:", graph_columns[:10])
        print("First 10 data columns:", col[:10])
        return
    
    # Load ensemble model trained by train_final_ensemble.py
    print("Loading ensemble model ..")
    model, config, scale, feature_names = load_ensemble_model(model_file, config_file, device)
    
    if model is None:
        print("CRITICAL ERROR: Failed to load ensemble model. Cannot proceed with forecasting.")
        print("Please check:")
        print(f"1. Model file exists: {os.path.exists(model_file)}")
        print(f"2. Model file path: {model_file}")
        print(f"3. Config file exists: {os.path.exists(config_file) if config_file else 'N/A'}")
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
    print(f"Forecast period: January 2025 - December 2027 (36 months)")
    
    # Save forecast results only for graph columns
    save_data_ensemble(full_data_tensor, forecast, confidence, graph_columns, index)
    
    # Apply smoothing for visualization (only for graph columns)
    all_combined = torch.cat((full_data_tensor, forecast), dim=0)
    
    # Global normalization for consistent scaling
    max_val = torch.max(all_combined)
    
    all_normalized = all_combined / max_val
    confidence_normalized = confidence / max_val
    
    # Apply exponential smoothing only to graph columns
    smoothed_data = torch.zeros_like(all_normalized)
    for col_name in graph_columns:
        if col_name in index:
            i = index[col_name]
            smoothed_col = exponential_smoothing(all_normalized[:, i].tolist(), 0.1)
            smoothed_data[:, i] = torch.FloatTensor(smoothed_col)
    
    smoothed_confidence = torch.zeros_like(confidence_normalized)
    for col_name in graph_columns:
        if col_name in index:
            i = index[col_name]
            smoothed_conf = exponential_smoothing(confidence_normalized[:, i].tolist(), 0.1)
            smoothed_confidence[:, i] = torch.FloatTensor(smoothed_conf)
    
    # Split back into historical and forecast periods
    historical_smoothed = smoothed_data[:-36]
    forecast_smoothed = smoothed_data[-36:]
    confidence_smoothed = smoothed_confidence
    
    # Generate plots for each attack in the graph
    print("Generating forecast plots...")
    plotted_count = 0
    for attack, solutions in graph.items():
        if attack in index:  # Only plot if attack exists in data
            plot_forecast_ensemble(historical_smoothed, forecast_smoothed, confidence_smoothed, 
                                 attack, solutions.copy(), index)
            save_gap_ensemble(forecast_smoothed, attack, solutions.copy(), index)
            plotted_count += 1
        else:
            print(f"Skipping '{attack}' - not found in data columns")
    
    print(f"Forecast completed successfully! Generated {plotted_count} plots.")
    print("Results saved in model/Ensemble/forecast/")
    print("Forecast period: January 2025 - December 2027")

if __name__ == "__main__":
    main()