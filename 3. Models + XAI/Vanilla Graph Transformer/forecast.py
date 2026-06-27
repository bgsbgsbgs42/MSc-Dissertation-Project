import pickle
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
import json
import pandas as pd
from datetime import datetime, timedelta
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric.transforms as T

from train import SimpleGraphTransformer, SimpleDataLoader

pyplot.rcParams['savefig.dpi'] = 1200

def _to_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, '%Y-%m-%d')
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).to_pydatetime()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return None


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
        name=name.replace('IZ', 'IS')# applicable only in this data (British English)
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
def zero_negative_curves(data, forecast, attack, solutions, index, col):

    #  Helper: normalize names like in plot_forecast() 
    def normalize(name):
        return name.strip().replace('/', '_').replace('\\', '_').replace(' ', '_')

    # Ensure everything is normalized consistently
    col = [normalize(c) for c in col]
    if isinstance(attack, str):
        attack = [attack]
    attack = [normalize(a) for a in attack]
    solutions = [normalize(s) for s in solutions]

    # Normalize index keys
    if isinstance(index, dict):
        index = {normalize(k): v for k, v in index.items()}

    valid_col_count = data.shape[1]
    print(f"[DEBUG] zero_negative_curves: {valid_col_count} columns available")

    #  Ensure all indices are within data.shape bounds 
    safe_index = {k: v for k, v in index.items() if v < valid_col_count}
    dropped = set(index.keys()) - set(safe_index.keys())
    if dropped:
        print(f" Dropping out-of-range indices: {dropped}")
    index = safe_index

    #  Process each attack 
    for atk in attack:
        if atk not in index:
            print(f" Skipping unmatched attack: {atk}")
            continue
        i = index[atk]
        a = data[:, i]
        f = forecast[:, i]

        # Zero out negative values safely
        a[a < 0] = 0
        f[f < 0] = 0

        data[:, i] = a
        forecast[:, i] = f

    #  Process each solution 
    for s in solutions:
        if s not in index:
            print(f" Skipping unmatched solution: {s}")
            continue
        i = index[s]
        a = data[:, i]
        f = forecast[:, i]

        # Zero out negative values safely
        a[a < 0] = 0
        f[f < 0] = 0

        data[:, i] = a
        forecast[:, i] = f

    print(f" zero_negative_curves complete: "
          f"{len(attack)} attacks | {len(solutions)} solutions processed")

    return data, forecast

    
            
        

#plots forecast of attack and relevant solutions trends. If alarming is set to True, plots the solutions trend forecasted to be less than the attack trend.
def plot_forecast(data, forecast, confidence, attack, solutions, index, col, alarming=True):

    #  Helper: Normalize names consistently 
    def normalize(name):
        return name.strip().replace('/', '_').replace('\\', '_').replace(' ', '_')

    # Normalize everything to prevent mismatched keys
    col = [normalize(c) for c in col]
    if isinstance(attack, str):
        attack = [attack]
    attack = [normalize(a) for a in attack]
    solutions = [normalize(s) for s in solutions]

    # Normalize index dictionary keys
    if isinstance(index, dict):
        index = {normalize(k): v for k, v in index.items()}

    print(f" Available columns: {len(col)} | Normalized index keys: {len(index)}")

    #  Defensive call to zero_negative_curves() 
    data, forecast = zero_negative_curves(data, forecast, attack, solutions, index, col)

    #  Plot setup 
    colours = [
        "RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
        "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
        "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
        "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
        "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
        "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"
    ]

    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    counter = 0

    #  Plot each attack 
    for atk in attack:
        if atk not in index:
            print(f"Skipping unmatched attack: {atk}")
            continue

        # Join past data and forecast
        d = torch.cat((data[:, index[atk]], forecast[0:1, index[atk]]), dim=0)
        f = forecast[:, index[atk]]
        c = confidence[:, index[atk]]
        label = atk

        ax.plot(range(len(d)), d, '-', color=colours[counter], label=label, linewidth=2)
        ax.plot(range(len(d)-1, (len(d)+len(f))-1), f, '-', color=colours[counter], linewidth=2)
        ax.fill_between(
            range(len(d)-1, (len(d)+len(f))-1),
            f - c, f + c,
            color=colours[counter], alpha=0.6
        )

        f_attack = f.clone()
        counter += 1

    #  Remove irrelevant solutions if alarming mode is on 
    if alarming and 'f_attack' in locals():
        solutions = [
            s for s in solutions
            if s in index and torch.mean(forecast[:, index[s]]) < torch.mean(f_attack)
        ]

    #  Plot solutions 
    for s in solutions:
        if s not in index:
            print(f" Skipping unmatched solution: {s}")
            continue

        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        label = s

        ax.plot(range(len(d)), d, '-', color=colours[counter], label=label, linewidth=1)
        ax.plot(range(len(d)-1, (len(d)+len(f))-1), f, '-', color=colours[counter], linewidth=1)
        ax.fill_between(
            range(len(d)-1, (len(d)+len(f))-1),
            f - c, f + c,
            color=colours[counter], alpha=0.6
        )

        # Highlight difference between attack and solution trends
        if torch.mean(f_attack) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, attack, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(
                    range(len(d)-1, (len(d)+len(f))-1),
                    cc - cc_conf, f + c,
                    color=colours[counter], alpha=0.3
                )
            else:
                # No comparable curve found — skip gap highlight
                pass
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, attack, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(
                    range(len(d)-1, (len(d)+len(f))-1),
                    cc + cc_conf, f - c,
                    color=colours[counter], alpha=0.3
                )
            else:
                # No comparable curve found — skip gap highlight
                pass
        counter += 1

    #  X-axis, labels, etc. 
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
             '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028']
    ax.set_xticks([6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174, 186, 198], years)
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(f"{attack}", y=1.03, fontsize=18)

    #  Save & show 
    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)
    images_dir = 'model/SimpleGraphTransformer/plots/'
    os.makedirs(images_dir, exist_ok=True)
    first_attack = attack[0] if len(attack) > 0 else "unknown_attack"
    pyplot.savefig(images_dir + first_attack + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + first_attack + ".pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


#saves the numerical forecast to text file as well as past data of each node
def save_data(data, forecast, confidence, variance, col):
    # write the data and forecast
    file_dir = 'model/SimpleGraphTransformer/forecast_data/'
    os.makedirs(file_dir, exist_ok=True)
    for i in range(data.shape[1]):
        d= data[:,i]
        f= forecast[:,i]
        c=confidence[:,i]
        v=variance[:,i]
        name=col[i]
        with open(file_dir+name.replace('/','_')+'.txt', 'w') as ff:
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n')
    ff.close()

#saves the forecasted trend's gap between attack and its relevant solutions to a csv file. The gap is for 3 years resulting in 3 values per solution.
def save_gap(forecast, attack, solutions,index, forecast_start_year=2025):
	# write the data and forecast
	gap_dir = 'model/SimpleGraphTransformer/forecast_gap/'
	os.makedirs(gap_dir, exist_ok=True)

	# Normalize keys like create_columns (replace '/' -> '_')
	def _norm_key(name):
		return name.strip().replace('/', '_')

	attack_key = _norm_key(attack)
	if attack_key not in index:
		print(f" save_gap: attack '{attack}' not found in index — skipping gap file.")
		return

	# Normalize and filter solutions
	norm_sols = [_norm_key(s) for s in solutions]
	valid_sols = [s for s in norm_sols if s in index]
	skipped = len(norm_sols) - len(valid_sols)
	if skipped:
		print(f" save_gap: skipped {skipped} unmatched solutions for attack '{attack}'.")

	with open(os.path.join(gap_dir, f"{_norm_key(consistent_name(attack))}_gap.csv"), 'w', newline='') as file:
		writer = csv.writer(file)
		a = forecast[:, index[attack_key]].tolist()
		a_reduced = [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]
		header_years = [str(forecast_start_year + i) for i in range(len(a_reduced))]
		writer.writerow(['Solution', *header_years])
		table = []

		a = forecast[:, index[attack_key]].tolist()
		a_reduced = [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]

		for s in valid_sols:
			row = [consistent_name(s)]
			f = forecast[:, index[s]].tolist()
			f_reduced = [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]
			gap = [x - y for x, y in zip(a_reduced, f_reduced)]
			row.extend(gap)
			table.append(row)

		if table:
			sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
			for row in sorted_table:
				writer.writerow(row)
    



#given data file, returns the list of column names and dictionary of the format (column name,column index)
def create_columns(file_name):
    col_name = []
    col_index = {}

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        col_name = [c.strip().replace('/', '_') for c in next(reader)]
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
            adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')
    return graph


def prepare_forecast_data(data_loader, forecast_start_date, forecast_months=36):
    """Prepare the input data for forecasting starting from a specific date"""
    
    # Convert forecast_start_date to datetime
    if isinstance(forecast_start_date, str):
        forecast_start_date = datetime.strptime(forecast_start_date, '%Y-%m-%d')
    
    # Find the index in dates that matches the forecast start date
    forecast_start_idx = None
    for i, date in enumerate(data_loader.dates):
        if date.year == forecast_start_date.year and date.month == forecast_start_date.month:
            forecast_start_idx = i
            break
    
    if forecast_start_idx is None:
        # Use the last available data point if exact date not found
        forecast_start_idx = len(data_loader.normalized_data) - data_loader.seq_len
        print(f"Warning: Forecast start date {forecast_start_date} not found in data. Using last available data.")
    
    # Get the sequence for forecasting
    seq_start = forecast_start_idx - data_loader.seq_len
    if seq_start < 0:
        seq_start = 0
        print("Warning: Not enough historical data, using available data from start")
    
    seq_data = data_loader.normalized_data[seq_start:forecast_start_idx]
    
    # If sequence is shorter than required, pad with zeros
    if len(seq_data) < data_loader.seq_len:
        padding = np.zeros((data_loader.seq_len - len(seq_data), data_loader.m))
        seq_data = np.vstack([padding, seq_data])
    
    return torch.FloatTensor(seq_data)


def forecast_with_graph_transformer(model, data_loader, forecast_start_date, forecast_months, device='cpu'):
    """Generate forecasts using the trained Graph Transformer model across the requested horizon."""
    if isinstance(forecast_start_date, str):
        forecast_start_dt = datetime.strptime(forecast_start_date, '%Y-%m-%d')
    else:
        forecast_start_dt = forecast_start_date

    normalized_buffer = np.array(data_loader.normalized_data, copy=True)
    date_sequence = getattr(data_loader, 'dates', None)
    date_list = []
    if date_sequence is not None:
        for value in date_sequence:
            dt = _to_datetime(value)
            if dt:
                date_list.append(dt)

    forecast_start_idx = None
    for idx, dt in enumerate(date_list):
        if dt.year == forecast_start_dt.year and dt.month == forecast_start_dt.month:
            forecast_start_idx = idx
            break
        if dt > forecast_start_dt:
            forecast_start_idx = idx
            break
    if forecast_start_idx is None:
        forecast_start_idx = normalized_buffer.shape[0]
        print(f"Warning: Forecast start date {forecast_start_dt.date()} not found in data. Using last available data point.")

    def build_sequence(start_idx):
        seq_start = start_idx - data_loader.seq_len
        if seq_start < 0:
            seq_start = 0
            print("Warning: Not enough historical data, using available data from start")
        seq = normalized_buffer[seq_start:start_idx]
        if seq.shape[0] < data_loader.seq_len:
            padding = np.zeros((data_loader.seq_len - seq.shape[0], normalized_buffer.shape[1]))
            seq = np.vstack([padding, seq])
        return torch.FloatTensor(seq)

    remaining = forecast_months
    current_start_idx = forecast_start_idx
    chunks_Y, chunks_conf, chunks_var = [], [], []
    scale_vec = data_loader.scale.detach().cpu().float()
    safe_scale = torch.where(scale_vec == 0, torch.ones_like(scale_vec), scale_vec)

    while remaining > 0:
        X_seq = build_sequence(current_start_idx)
        forecast_data_list = data_loader.prepare_pyg_data(
            X_seq.unsqueeze(0).numpy(),
            np.zeros((1, data_loader.out_len, data_loader.m))
        )
        if not forecast_data_list:
            raise ValueError("Failed to prepare forecast data")

        outputs = []
        with torch.no_grad():
            batch = forecast_data_list[0].to(device)
            output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            outputs.append(output.squeeze(0).cpu())

        outputs = torch.stack(outputs)
        chunk_mean = torch.mean(outputs, dim=0)
        chunk_var = torch.var(outputs, dim=0, unbiased=False)
        chunk_std = torch.sqrt(chunk_var)

        num_samples = outputs.size(0) if outputs.dim() > 0 else 1
        z = 1.96
        denom = torch.sqrt(torch.tensor(float(num_samples), dtype=chunk_std.dtype))
        chunk_conf = z * chunk_std / denom

        chunk_Y_scaled = chunk_mean * scale_vec
        chunk_conf_scaled = chunk_conf * scale_vec
        chunk_var_scaled = chunk_var * (scale_vec ** 2)

        if chunk_Y_scaled.shape[0] == 0:
            raise ValueError("Model returned empty forecast chunk; cannot continue.")

        take = min(chunk_Y_scaled.shape[0], remaining)
        chunks_Y.append(chunk_Y_scaled[:take])
        chunks_conf.append(chunk_conf_scaled[:take])
        chunks_var.append(chunk_var_scaled[:take])

        chunk_norm = chunk_Y_scaled[:take] / safe_scale
        zero_mask = scale_vec == 0
        if zero_mask.any():
            chunk_norm[:, zero_mask] = 0
        normalized_buffer = np.vstack([normalized_buffer, chunk_norm.numpy()])

        current_start_idx += take
        remaining -= take

    Y_scaled = torch.cat(chunks_Y, dim=0)
    confidence_scaled = torch.cat(chunks_conf, dim=0)
    variance_scaled = torch.cat(chunks_var, dim=0)

    return Y_scaled, confidence_scaled, variance_scaled


#This script forecasts the future of the graph, up to 3 years in advance

def main():
    # Configuration - adjustable parameters
    data_file = './data/sm_data_g.csv'  
    model_file = 'model/SimpleGraphTransformer/final_model.pt'
    model_info_file = 'model/SimpleGraphTransformer/final_model_info.json'
    nodes_file = 'data/sm_data_g.csv'  
    graph_file = 'data/graph.csv'
    
    # Adjustable forecast parameters
    forecast_start_date = '2025-01-01'  # Adjustable parameter
    forecast_start_dt = datetime.strptime(forecast_start_date, '%Y-%m-%d')
    forecast_months = 36  # 3 years: 2025-01-01 to 2027-12-31
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model info to get hyperparameters
    try:
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)
        hp = model_info['hyperparameters']
        print('Loaded model hyperparameters:', hp)
    except FileNotFoundError:
        print(f"Warning: {model_info_file} not found. Using default hyperparameters.")
        hp = {
            'channels': 64,
            'num_layers': 3,
            'pe_dim': 8,
            'pe_walk_length': 20,
            'node_dim': 1,
            'num_heads': 4,
            'attn_dropout': 0.2,
            'dropout': 0.3,
            'weight_decay': 1e-4,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'forecast_horizon': 36,
            'sequence_length': 18,
            'correlation_threshold': 0.3,
            'local_gnn_type': 'GINE'
        }
    
    # Load data using the same SimpleDataLoader as training
    print('Loading data...')
    try:
        data_loader = SimpleDataLoader(
            data_file, 
            device, 
            seq_len=hp['sequence_length'], 
            out_len=hp['forecast_horizon'],
            correlation_threshold=hp['correlation_threshold']
        )
        hp['num_nodes'] = data_loader.m
        print(f"Data loaded successfully! Number of nodes: {data_loader.m}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create and load model
    print('Creating model...')
    model = SimpleGraphTransformer(hp).to(device)
    
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"Model loaded from: {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Load column names and graph structure
    col, index = create_columns(nodes_file)
    graph = build_graph(graph_file)
    
    # Generate forecasts
    print(f"Generating forecasts from {forecast_start_date} for {forecast_months} months...")
    Y, confidence, variance = forecast_with_graph_transformer(
        model, data_loader, forecast_start_date, forecast_months, device
    )
    forecast_months = Y.shape[0]
    print(f'Forecast shape: {Y.shape}')
    
    # Prepare historical data for plotting
    historical_data = torch.from_numpy(data_loader.raw_data).float()
    history_cutoff_idx = historical_data.shape[0]
    date_sequence = getattr(data_loader, 'dates', None)
    seq_list = None
    if date_sequence is not None:
        try:
            seq_list = list(date_sequence)
        except TypeError:
            seq_list = None
    if seq_list:
        for idx, date_value in enumerate(seq_list):
            dt_value = _to_datetime(date_value)
            if dt_value and dt_value >= forecast_start_dt:
                history_cutoff_idx = idx
                break
    if history_cutoff_idx == 0:
        history_cutoff_idx = historical_data.shape[0]
    historical_for_plot = historical_data[:history_cutoff_idx]
    
    # Save results
    save_data(historical_data, Y, confidence, variance, col)
    
    # Combine historical data and forecast for plotting
    all_data = torch.cat((historical_for_plot, Y), dim=0)
    
    # Scale down for visualization (global normalization)
    incident_max = -999999999
    mention_max = -999999999
    
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16,32):
                continue
            if 'Mention' in col[j]:
                if all_data[i,j] > mention_max:
                    mention_max = all_data[i,j]
            else:
                if all_data[i,j] > incident_max:
                    incident_max = all_data[i,j]
    
    all_normalized = torch.zeros(all_data.shape[0], all_data.shape[1])
    confidence_normalized = torch.zeros(confidence.shape[0], confidence.shape[1])
    
    u = 0
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if 'Mention' in col[j]:
                all_normalized[i,j] = all_data[i,j] / mention_max
            else:
                all_normalized[i,j] = all_data[i,j] / incident_max
            
            if i >= all_data.shape[0] - forecast_months:
                if all_data[i,j] != 0:  # Avoid division by zero
                    confidence_normalized[u,j] = confidence[u,j] * (all_normalized[i,j] / all_data[i,j])
        if i >= all_data.shape[0] - forecast_months:
            u += 1
    
    # Apply smoothing
    smoothed_data = torch.stack([torch.FloatTensor(exponential_smoothing(all_normalized[:, i].tolist(), 0.1)) 
                               for i in range(all_normalized.shape[1])], dim=1)
    smoothed_confidence = torch.stack([torch.FloatTensor(exponential_smoothing(confidence_normalized[:, i].tolist(), 0.1)) 
                                     for i in range(confidence_normalized.shape[1])], dim=1)
    
    # Plot forecasts for each attack in the graph
    for attack, solutions in graph.items():
        plot_forecast(smoothed_data[:-forecast_months,], smoothed_data[-forecast_months:,], 
                     smoothed_confidence, attack, solutions, index, col)
        save_gap(smoothed_data[-forecast_months:,], attack, solutions, index, forecast_start_year=forecast_start_dt.year)
    
    print("Forecast completed successfully!")


if __name__ == "__main__":
    main()