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
import pandas as pd
from datetime import datetime, timedelta

# Import the model class from train.py
from train import SimpleVisionTransformer
pyplot.rcParams['savefig.dpi'] = 1200


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
    print(f" zero_negative_curves: {valid_col_count} columns available")

    #  Ensure all indices are within data.shape bounds 
    safe_index = {k: v for k, v in index.items() if v < valid_col_count}
    dropped = set(index.keys()) - set(safe_index.keys())
    if dropped:
        print(f"Dropping out-of-range indices: {dropped}")
    index = safe_index

    #  Process each attack 
    for atk in attack:
        if atk not in index:
            print(f"Skipping unmatched attack: {atk}")
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
            print(f"Skipping unmatched solution: {s}")
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
            ax.fill_between(
                range(len(d)-1, (len(d)+len(f))-1),
                cc - cc_conf, f + c,
                color=colours[counter], alpha=0.3
            )
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, attack, solutions, col)
            ax.fill_between(
                range(len(d)-1, (len(d)+len(f))-1),
                cc + cc_conf, f - c,
                color=colours[counter], alpha=0.3
            )
        counter += 1

    #  X-axis, labels, etc. 
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
             '2020', '2021', '2022', '2023', '2024', '2025', '2026']
    ax.set_xticks([6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174], years)
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title("Forecast Plot", y=1.03, fontsize=18)

    #  Save & show 
    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)
    images_dir = 'model/Bayesian/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    first_attack = attack[0] if len(attack) > 0 else "unknown_attack"
    safe_name = first_attack.replace('/', '_').replace('\\', '_')
    pyplot.savefig(os.path.join(images_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.savefig(os.path.join(images_dir, safe_name + ".pdf"), bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


#saves the numerical forecast to text file as well as past data of each node
def save_data(data, forecast, confidence, variance, col):
    # write the data and forecast
    file_dir = 'model/Bayesian/forecast/data/'
    os.makedirs(file_dir, exist_ok=True)
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        with open(os.path.join(file_dir, name.replace('/', '_') + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')


#saves the forecasted trend's gap between attack and its relevant solutions to a csv file. The gap is for 3 years resulting in 3 values per solution.
def save_gap(forecast, attack, solutions,index):
	# write the data and forecast
	os.makedirs('model/Bayesian/forecast/gap/', exist_ok=True)

	def _norm_key(n):
		return n.strip().replace('/', '_')

	attack_key = _norm_key(attack)
	if attack_key not in index:
		print(f"save_gap: attack '{attack}' not found in index — skipping gap file.")
		return

	# Normalize solutions and filter those present in index
	norm_sols = [_norm_key(s) for s in solutions]
	valid_sols = [s for s in norm_sols if s in index]
	skipped = len(norm_sols) - len(valid_sols)
	if skipped:
		print(f" save_gap: skipped {skipped} unmatched solutions for attack '{attack}'.")

	out_file = os.path.join('model/Bayesian/forecast/gap/', f"{_norm_key(consistent_name(attack))}_gap.csv")
	with open(out_file, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Solution', '2023', '2024', '2025'])
		table = []

		a = forecast[:, index[attack_key]].tolist()
		a_reduced = [sum(a[i:i + 12]) / 12 for i in range(0, len(a), 12)]

		for s in valid_sols:
			row = [consistent_name(s)]
			f = forecast[:, index[s]].tolist()
			f_reduced = [sum(f[i:i + 12]) / 12 for i in range(0, len(f), 12)]
			gap = [x - y for x, y in zip(a_reduced, f_reduced)]
			row.extend(gap)
			table.append(row)

		if table:
			for row in sorted(table, key=lambda r: sum(r[-3:])):
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


# Load the ViT model trained by train.py
def load_vit_model(model_file, device):
    """Load the ViT model trained by train.py"""
    print(f"Loading ViT model from: {model_file}")
    
    # Load the saved model data
    model_data = torch.load(model_file, map_location=device, weights_only=False)
    
    # Extract hyperparameters and model state
    hp = model_data['hyperparameters']
    
    
    # Create model instance
    model = SimpleVisionTransformer(hp).to(device)
    
    # Load state dict
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully with {sum(p.nelement() for p in model.parameters())} parameters")
    return model, hp

# Generate forecast dates
def generate_forecast_dates(start_date, months=36):
    """Generate monthly dates for forecast period"""
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(months):
        dates.append(current.strftime('%Y-%m-%d'))
        # Add approximately one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return dates

#This script forecasts the future of the graph, up to 3 years in advance

def main():
    # Configurable parameters
    data_file = './data/sm_data.txt'
    model_file = 'model/ViT_Final/o_model.pt'  
    nodes_file = 'data/data.csv'
    graph_file = 'data/graph.csv'
    
    # Adjustable forecast start date (default: 2025-01-01)
    forecast_start_date = '2025-01-01'  # Can make this a command line argument
    
    # Forecast period (3 years = 36 months)
    forecast_months = 36
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #read the data
    fin = open(data_file)
    rawdat = np.loadtxt(fin, delimiter='\t')
    n, m = rawdat.shape

    #load column names and dictionary of (column name, index)
    col, index = create_columns(nodes_file)

    #build the graph in the format {attack:list of pertinent technologies}
    graph = build_graph(graph_file)

    #for normalisation
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)

    #normalise
    for i in range(m):
        scale[i] = np.max(np.abs(rawdat[:, i]))
        dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))
        
    print('data shape:', dat.shape)

    # Load the ViT model
    model, hp = load_vit_model(model_file, device)
    
    # Prepare input data for forecasting
    sequence_length = hp['sequence_length']
    forecast_horizon = hp['forecast_horizon']
    
    print(f"Model sequence length: {sequence_length}")
    print(f"Model forecast horizon: {forecast_horizon}")
    
    # Use the last sequence_length months as input
    X = torch.from_numpy(dat[-sequence_length:, :]).float().to(device)
    X = X.unsqueeze(0)  # Add batch dimension
    
    print(f"Input shape: {X.shape}")
    
    outputs = []

    # Generate forecasts
    print("Generating forecasts...")
    with torch.no_grad():
        for _ in range(1):
            output = model(X)
            # Take the forecast horizon from the output
            y_pred = output[0, :, :].clone()  # (forecast_horizon, num_nodes)
            outputs.append(y_pred)

    # Stack and process outputs
    outputs = torch.stack(outputs)  # (num_runs, forecast_horizon, num_nodes)
    
    # If we need more than forecast_horizon months, use recursive forecasting
    if forecast_months > forecast_horizon:
        print(f"Recursive forecasting for {forecast_months} months...")
        all_outputs = []
        
        for run in range(1):
            current_input = X.clone()
            run_outputs = []
            
            for step in range(0, forecast_months, forecast_horizon):
                with torch.no_grad():
                    output = model(current_input)
                    step_forecast = output[0, :, :].clone()  # (forecast_horizon, num_nodes)
                    run_outputs.append(step_forecast)
                
                # Update input for next step (use the forecast as new input)
                if step + forecast_horizon < forecast_months:
                    # Remove oldest data and append forecast
                    new_input = torch.cat([
                        current_input[:, forecast_horizon:, :], 
                        output
                    ], dim=1)
                    current_input = new_input
            
            # Concatenate all steps for this run
            run_forecast = torch.cat(run_outputs, dim=0)[:forecast_months]
            all_outputs.append(run_forecast)
        
        outputs = torch.stack(all_outputs)  # (num_runs, forecast_months, num_nodes)

    # Calculate statistics
    Y = torch.mean(outputs, dim=0)  # (forecast_months, num_nodes)
    variance = torch.var(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)
    
    # Calculate 95% confidence interval
    z = 1.96
    confidence = z * std_dev / torch.sqrt(torch.tensor(1))

    # Denormalize
    scale_tensor = torch.from_numpy(scale).float().to(device)
    Y = Y * scale_tensor
    variance = variance * scale_tensor
    confidence = confidence * scale_tensor

    print(f'Forecast shape: {Y.shape}')

    # Prepare data for plotting and saving
    dat_tensor = torch.from_numpy(rawdat).float()
    
    # Save the data
    save_data(dat_tensor, Y, confidence, variance, col)

    # Combine historical data with forecast for plotting
    all_data = torch.cat((dat_tensor, Y), dim=0)

    # Scale down full data (global normalisation) for plotting
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

    all_n = torch.zeros(all_data.shape[0], all_data.shape[1])
    confidence_n = torch.zeros(confidence.shape[0], confidence.shape[1])
    u = 0
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if 'Mention' in col[j]:
                all_n[i,j] = all_data[i,j] / mention_max
            else:
                all_n[i,j] = all_data[i,j] / incident_max
            
            if i >= all_data.shape[0] - forecast_months:
                if all_data[i,j] != 0:  # Avoid division by zero
                    confidence_n[u,j] = confidence[u,j] * (all_n[i,j] / all_data[i,j])
        if i >= all_data.shape[0] - forecast_months:
            u += 1

    # Apply smoothing
    # Apply smoothing per column (each column is a 1-D time series)
    smoothed_cols = []
    for j in range(all_n.shape[1]):
        col_series = all_n[:, j].tolist()
        smoothed_col = exponential_smoothing(col_series, 0.1)
        smoothed_cols.append(torch.FloatTensor(smoothed_col))
    # smoothed_dat shape: (time_steps, num_nodes)
    smoothed_dat = torch.stack(smoothed_cols, dim=1)

    # Smooth confidence per column (confidence_n has shape: forecast_months x num_nodes)
    conf_cols = []
    for j in range(confidence_n.shape[1]):
        conf_series = confidence_n[:, j].tolist()
        conf_smoothed_col = exponential_smoothing(conf_series, 0.1)
        conf_cols.append(torch.FloatTensor(conf_smoothed_col))
    smoothed_confidence = torch.stack(conf_cols, dim=1)

    # Plot all forecasted nodes in the graph as groups of plots
    for attack, solutions in graph.items():
        plot_forecast(smoothed_dat[:-forecast_months,], smoothed_dat[-forecast_months:,], 
                     smoothed_confidence, attack, solutions, index, col)
        save_gap(smoothed_dat[-forecast_months:,], attack, solutions, index)

    print(f"Forecast completed for period: {forecast_start_date} to {generate_forecast_dates(forecast_start_date, forecast_months)[-1]}")
    print(f"Generated {forecast_months} months of forecasts")

if __name__ == "__main__":
    main()