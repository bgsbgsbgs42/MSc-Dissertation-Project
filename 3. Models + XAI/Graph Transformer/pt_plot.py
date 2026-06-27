import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random
import re
import pandas as pd
from datetime import datetime, timedelta
from torch_geometric.data import Data


pyplot.rcParams['savefig.dpi'] = 1200

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

def sanitize_filename(name):
    """Remove or replace invalid filename characters"""
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Remove any other non-printable characters
    name = re.sub(r'[^\x20-\x7E]', '', name)
    return name

def zero_negative_curves(data, forecast, s, index):
    """Negative values (due to smoothing) are changed to 0"""
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

def plot_forecast_with_dates(data, forecast, confidence, s, index, col, dates, forecast_start_date):
    """Plot past data and forecast of a single pertinent technology node s with actual dates"""
    
    # Skip if technology not in data
    if s not in index:
        print(f"Warning: Technology '{s}' not found in data. Skipping plot.")
        return
    
    data, forecast = zero_negative_curves(data, forecast, s, index)
    
     
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Generate forecast dates
    forecast_dates = generate_forecast_dates(forecast_start_date, len(forecast))
    
    # Get the data for this technology
    historical_data = data[:, index[s]]
    forecast_data = forecast[:, index[s]]
    confidence_data = confidence[:, index[s]]
    
    # Connect historical and forecast data for continuous line
    d = torch.cat((historical_data, forecast_data[0:1]), dim=0)
    f = forecast_data
    c = confidence_data
    
    s_name = consistent_name(s)
    
    # Create time indices
    historical_time = list(range(len(historical_data)))
    forecast_time = list(range(len(historical_data), len(historical_data) + len(forecast_data)))
    
    # Plot historical data
    ax.plot(historical_time, historical_data, '-', color='red', label=f'{s_name} (Historical)', linewidth=1)
    
    # Plot forecast with confidence interval
    ax.plot(forecast_time, f, '-', color='red', linewidth=1)
    ax.fill_between(forecast_time, f - c, f + c, color='red', alpha=0.6)
    
    # Set x-axis labels
    try:
        # Create year labels from dates
        year_labels = []
        year_positions = []
        
        # Use the first date as reference
        if dates:
            # Try to extract years from dates
            for i, date in enumerate(dates):
                if isinstance(date, str) and '-' in date:
                    year = date.split('-')[0]
                    if i % 12 == 6:  # Mark mid-year
                        year_labels.append(year)
                        year_positions.append(i)
        
        # If we couldn't extract years, use generic labels
        if not year_labels:
            # Generate labels based on forecast start date
            start_year = int(forecast_start_date.split('-')[0])
            years = list(range(start_year - 5, start_year + 4))  # Show some historical years
            year_labels = [str(year) for year in years[-10:]]  # Last 10 years
            year_positions = [i * 12 + 6 for i in range(len(year_labels))]
        
        ax.set_xticks(year_positions)
        ax.set_xticklabels(year_labels)
        
    except Exception as e:
        print(f"Warning: Could not set date labels: {e}")
        # Fallback to generic labels
        x = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']
        positions = [6 + 12 * i for i in range(len(x))]
        ax.set_xticks(positions)
        ax.set_xticklabels(x)

    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10})
    ax.axis('tight')
    ax.grid(True, alpha=0.3)
    pyplot.xticks(rotation=45, fontsize=10)
    
    # Add title with date ranges
    historical_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"
    forecast_range = f"{forecast_dates[0]} to {forecast_dates[-1]}" if forecast_dates else "N/A"
    
    pyplot.title(f"{s_name}\nHistorical: {historical_range} | Forecast: {forecast_range}", 
                 y=1.03, fontsize=14)

    # Add vertical line to separate historical and forecast periods
    separation_point = len(historical_data)
    ax.axvline(x=separation_point - 0.5, color='gray', linestyle='--', alpha=0.7)
    ax.text(separation_point + 5, ax.get_ylim()[1] * 0.9, 'Forecast\nPeriod', 
            rotation=0, verticalalignment='top', fontsize=10, ha='center')

    fig = pyplot.gcf()
    fig.set_size_inches(12, 8)

    # Save and show the forecast
    images_dir = 'model/GraphTransformer/forecast/pt_plots/'
    os.makedirs(images_dir, exist_ok=True)
    safe_filename = sanitize_filename(s_name)
    pyplot.savefig(images_dir + safe_filename + '.png', bbox_inches="tight", dpi=300)
    pyplot.savefig(images_dir + safe_filename + ".pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(2)
    pyplot.close()

def create_columns(file_name):
    """Given data file, returns the list of column names and dictionary of the format (column name,column index)"""
    col_name = []
    col_index = {}

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
            adjacent_nodes = [node for node in row[1:] if node]  # does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with', len(graph), 'attacks...')
    return graph

def prepare_forecast_data(historical_data):
    """Prepare normalized data"""
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

def load_operational_model(model_path, device, num_nodes):
    """Load the operational model"""
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
    
    # Import GraphTransformer from your forecast_w_pretrain.py
    sys.path.append('.')
    from forecast_w_pretrain import GraphTransformer
    
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

# Main execution
if __name__ == '__main__':
    import argparse
    
    # Configuration with adjustable parameters
    parser = argparse.ArgumentParser(description='Generate pertinent technology forecasts using operational model')
    parser.add_argument('--forecast_start_date', type=str, default='2025-01-01', 
                       help='Start date for forecasting (format: YYYY-MM-DD)')
    parser.add_argument('--forecast_months', type=int, default=36,
                       help='Number of months to forecast')
    parser.add_argument('--data_file', type=str, default='Dissertation/Graph Transformer/data/sm_data_g.csv',
                       help='Path to smoothed data file')
    parser.add_argument('--model_file', type=str, default='Dissertation/Graph Transformer/model/GraphTransformer/o_model.pt',
                       help='Path to operational model file')
    parser.add_argument('--nodes_file', type=str, default='Dissertation/Graph Transformer/data/sm_data_g.csv',
                       help='Path to original data file with dates')
    parser.add_argument('--graph_file', type=str, default='Dissertation/Graph Transformer/data/graph.csv',
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
    
    # Collect all pertinent technologies from the graph
    all_pertinent_techs = set()
    for attack, solutions in graph.items():
        all_pertinent_techs.update(solutions)
    
    print(f"Found {len(all_pertinent_techs)} pertinent technologies in graph")
    
    # Filter to only include technologies that exist in historical data
    pertinent_techs_to_forecast = []
    for tech in all_pertinent_techs:
        if tech in historical_index:
            pertinent_techs_to_forecast.append(tech)
        else:
            print(f"Warning: Technology '{tech}' not found in historical data, skipping")
    
    print(f"Forecasting for {len(pertinent_techs_to_forecast)} pertinent technologies")
    
    # Create a subset of data containing only the technologies we care about
    relevant_columns = list(pertinent_techs_to_forecast)
    
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
    print("Creating graph structure...")
    edge_index, edge_attr = create_graph_structure(normalized_data, num_nodes=len(relevant_columns))
    
    # Load the operational model
    print("Loading operational model...")
    model, hp = load_operational_model(args.model_file, device, len(relevant_columns))
    
    # Generate forecast
    print(f"Generating {args.forecast_months}-month forecast...")
    mean_forecast_norm, std_forecast_norm, all_predictions = create_simple_forecast(
        model, normalized_data, edge_index, edge_attr, 
        seq_len=12, forecast_months=args.forecast_months, num_samples=50
    )
    
    # Denormalize the forecast back to original scale
    mean_forecast = denormalize_forecast(mean_forecast_norm, data_min, data_max, data_range)
    std_forecast = denormalize_forecast(std_forecast_norm, data_min, data_max, data_range)
    
    # Calculate confidence intervals (95%)
    z = 1.96
    confidence = z * std_forecast / np.sqrt(50)
    
    # Prepare data for plotting
    historical_plot = torch.from_numpy(relevant_data).float()
    forecast_plot = torch.from_numpy(mean_forecast).float()
    confidence_tensor = torch.from_numpy(confidence).float()
    
    # Apply minimal smoothing for visualization
    historical_smoothed = torch.zeros_like(historical_plot)
    forecast_smoothed = torch.zeros_like(forecast_plot)
    
    for j in range(historical_plot.shape[1]):
        # Light smoothing only
        hist_smooth = exponential_smoothing(historical_plot[:, j].numpy(), 0.03)
        fcst_smooth = exponential_smoothing(forecast_plot[:, j].numpy(), 0.03)
        historical_smoothed[:, j] = torch.tensor(hist_smooth)
        forecast_smoothed[:, j] = torch.tensor(fcst_smooth)
    
    # Normalize each column separately for better visualization
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
    
    # Plot forecasts for each pertinent technology
    print("Generating plots for pertinent technologies...")
    plotted_count = 0
    
    for tech in pertinent_techs_to_forecast:
        if tech in relevant_index:
            print(f"Plotting forecast for {tech}...")
            plot_forecast_with_dates(
                historical_normalized, forecast_normalized, confidence_tensor,
                tech, relevant_index, relevant_columns, dates, args.forecast_start_date
            )
            plotted_count += 1
    
    print(f"Successfully generated plots for {plotted_count} pertinent technologies")