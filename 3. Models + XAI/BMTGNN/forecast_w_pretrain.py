import pickle
import argparse
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
from datetime import datetime, timedelta
import pandas as pd

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
def zero_negative_curves(data, forecast, attack, solutions, index_map):
    """
    Replace negative values with zero for attack and solution series.
    index_map: dict mapping column name -> column index
    """
    # Attack
    if attack in index_map:
        a_idx = index_map[attack]
        a = data[:, a_idx]
        f = forecast[:, a_idx]
        # In-place clamp negatives to zero
        a[a < 0] = 0
        f[f < 0] = 0
    else:
        # attack not present -> no-op for attack
        pass

    # Solutions
    for s in solutions:
        if s in index_map:
            s_idx = index_map[s]
            a = data[:, s_idx]
            f = forecast[:, s_idx]
            a[a < 0] = 0
            f[f < 0] = 0
        else:
            # solution not present -> skip
            continue

    return data, forecast
           

#plots forecast of attack and relevant solutions trends. If alarming is set to True, plots the solutions trend forecasted to be less than the attack trend.
def plot_forecast(data,forecast,confidence,attack,solutions,index,col,alarming=True, forecast_start_year=2025):

    data,forecast= zero_negative_curves(data, forecast, attack, solutions, index)
    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
          "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
          "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
          "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
          "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
          "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])


    #Plot the forecast of attack
    counter=0
    d=torch.cat((data[:,index[attack]],forecast[0:1,index[attack]]),dim=0)#connect the past to future in the plot
    f=forecast[:,index[attack]]
    c=confidence[:,index[attack]]
    a=consistent_name(attack)
    ax.plot(range(len(d)),d,'-', color=colours[counter],label=a,linewidth = 2)
    ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=2)
    ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.6)
    f_attack=f.clone()
    counter+=1

    #remove technologies that we are not worried about in the future
    if alarming:
        for s in list(solutions):
            f=forecast[:,index[s]]
            if torch.mean(f)>= torch.mean(f_attack): #solution with higher trend in the future
                solutions.remove(s)

    #Plot the forecast of the solutions
    for s in solutions:
        d=torch.cat((data[:,index[s]],forecast[0:1,index[s]]),dim=0)#connect the past to future in the plot
        f=forecast[:,index[s]]
        c=confidence[:,index[s]]
        s=consistent_name(s)
        ax.plot(range(len(d)),d,'-', color=colours[counter],label=s,linewidth = 1)
        ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=1)
        ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.6)
        if torch.mean(f_attack) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, attack, solutions, col)  # to highlight the gap
            if cc is not None and cc_conf is not None:
                ax.fill_between(range(len(d)-1, (len(d)+len(f))-1), cc - cc_conf, f + c, color=colours[counter], alpha=0.3)
            else:
                # No comparable curve found — skip gap highlight
                pass
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, attack, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(range(len(d)-1, (len(d)+len(f))-1), cc + cc_conf, f - c, color=colours[counter], alpha=0.3)
            else:
                # No comparable curve found — skip gap highlight
                pass

        counter+=1  
    
    # Static x-axis labels: years from 2012 to 2028
    x = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028']
    # Positions: every 12 months starting at month 6
    ax_positions = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174, 186, 198]
    ax.set_xticks(ax_positions, x)

    #ax.axvspan(138, 173, color="skyblue", alpha=0.6,  label="Forecast Period")
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(a, y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7) 

    #save and show the forecast
    images_dir = 'Dissertation/Bayesian MTGNN/model/Bayesian/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    pyplot.savefig(images_dir+a.replace('/','_')+'.png', bbox_inches="tight")
    pyplot.savefig(images_dir+a.replace('/','_')+".pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


#saves the numerical forecast to text file as well as past data of each node
def save_data(data, forecast, confidence, variance, col, forecast_start_date):
    # Create directory if it doesn't exist
    file_dir = 'Dissertation/Bayesian MTGNN/model/Bayesian/forecast/data/'
    os.makedirs(file_dir, exist_ok=True)
    
    # write the data and forecast
    for i in range(data.shape[1]):
        d= data[:,i]
        f= forecast[:,i]
        c=confidence[:,i]
        v=variance[:,i]
        name=col[i]
        with open(file_dir+name.replace('/','_')+'.txt', 'w') as ff:
            ff.write(f'Forecast Start Date: {forecast_start_date}\n')
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n')
            ff.close()

#saves the forecasted trend's gap between attack and its relevant solutions to a csv file. The gap is for 3 years resulting in 3 values per solution.
def save_gap(forecast, attack, solutions,index, forecast_start_year):
    # Create directory if it doesn't exist
    os.makedirs('Dissertation/Bayesian MTGNN/model/Bayesian/forecast/gap/', exist_ok=True)
    
    # write the data and forecast
    with open('Dissertation/Bayesian MTGNN/model/Bayesian/forecast/gap/'+consistent_name(attack).replace('/','_')+'_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row
        writer.writerow(['Solution', str(forecast_start_year), str(forecast_start_year+1), str(forecast_start_year+2)])
        table=[]
        a=forecast[:,index[attack]].tolist()
        a_reduced= [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]#mean of every 12 months
        for s in solutions:
            row=[consistent_name(s)]
            f=forecast[:,index[s]].tolist()
            f_reduced= [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]#mean of every 12 months
            
            gap=[x - y for x, y in zip(a_reduced, f_reduced)]#calculate the gap
            row.extend(gap)#3 years gap
            table.append(row)
        sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
        for row in sorted_table:
            writer.writerow(row)
    




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
            col_name= col_name[1:]
        
        for i,c in enumerate(col_name):
            col_index[c]=i
        
        return col_name,col_index


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


def get_lookback_data(dat, forecast_start_date, seq_in_len=10):
    """
    Extract the lookback data based on forecast start date
    Returns data for the last 'seq_in_len' months before forecast_start_date
    """
    # Parse forecast start date
    forecast_date = datetime.strptime(forecast_start_date, '%Y-%m-%d')
    
    # Calculate the end date for lookback period (one month before forecast start)
    lookback_end = forecast_date - timedelta(days=1)
    
    
    lookback_data = dat[-seq_in_len:, :]
    
    print(f"Using lookback data from {seq_in_len} months before {forecast_start_date}")
    return lookback_data


#This script forecasts the future of the graph, up to 3 years in advance

def main(forecast_start_date = '2028-01-01'):
    data_file='Dissertation/Bayesian MTGNN/data/sm_data_g.csv'
    model_file='Dissertation/Bayesian MTGNN/model/o_model.pt'  # Updated path to match train_o_model.py
    nodes_file='Dissertation/Bayesian MTGNN/data/sm_data_g.csv'
    graph_file='Dissertation/Bayesian MTGNN/data/graph.csv'
    
    # Parse forecast parameters
    forecast_start_date = '2028-01-01'
    forecast_start_year = int(forecast_start_date.split('-')[0])
    seq_out_len = 36  # 3 years * 12 months = 36 months

    # Robust data load using pandas to handle headers, mixed delimiters, and non-numeric rows
    try:
        # Try to infer the delimiter
        df = pd.read_csv(data_file, sep=None, engine='python')
    except Exception:
        # Fallback to tab-delimited
        df = pd.read_csv(data_file, delimiter='\t', engine='python', header=0)

    # If file was loaded into a single column and the first cell contains comma-separated names, re-read using comma separator.
    if df.shape[1] == 1 and isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].count(',') > 0:
        try:
            df = pd.read_csv(data_file, sep=',', engine='python')
        except Exception:
            pass

    # If first column is a Date column, drop it (many files have Date in first column)
    first_col = str(df.columns[0]).lower()
    if 'date' in first_col:
        df = df.iloc[:, 1:]

    # Coerce all columns to numeric (non-numeric -> NaN) and fill NaNs sensibly
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

    rawdat = numeric_df.values.astype(float)
    n, m = rawdat.shape

    #load column names and dictionary of (column name, index)
    col,index=create_columns(nodes_file)


    #build the graph in the format {attack:list of pertinent technologies}
    graph=build_graph(graph_file)

    #for normalisation
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)

    #normalise
    for i in range(m):
        scale[i] = np.max(np.abs(rawdat[:, i]))
        dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))
        

    print('data shape:',dat.shape)

    #preparing last part of the data to be used for the forecast
    seq_in_len = 10 #look back
    X = get_lookback_data(dat, forecast_start_date, seq_in_len)
    X = torch.from_numpy(X) #look back months
    X = torch.unsqueeze(X,dim=0)
    X = torch.unsqueeze(X,dim=1)
    X = X.transpose(2,3)
    X = X.to(torch.float)


    #load the model - using state_dict loading to match train_o_model.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import the model architecture
    from hyperparameter_optim_bmtgnn_transfer_learning import gtnet
    
    # Load optimal hyperparameters to initialize model
    def load_hyperparameters(hp_file='Dissertation/Bayesian MTGNN/model/hp.txt'):
        with open(hp_file, 'r') as f:
            hp_str = f.read().strip()
        hp_list = eval(hp_str)
        
        hyperparams = {
            'gcn_depth': hp_list[0], 'lr': hp_list[1], 'conv_channels': hp_list[2],
            'residual_channels': hp_list[3], 'skip_channels': hp_list[4], 
            'end_channels': hp_list[5], 'subgraph_size': hp_list[6], 'dropout': hp_list[7],
            'dilation_exponential': hp_list[8], 'node_dim': hp_list[9], 
            'propalpha': hp_list[10], 'tanhalpha': hp_list[11], 'layers': hp_list[12]
        }
        return hyperparams

    hp = load_hyperparameters()
    
    # Initialize model with correct architecture
    num_nodes = m
    model = gtnet(
        gcn_true=True, buildA_true=True, gcn_depth=hp['gcn_depth'], 
        num_nodes=num_nodes, device=device, predefined_A=None,
        dropout=hp['dropout'], subgraph_size=hp['subgraph_size'],
        node_dim=hp['node_dim'], dilation_exponential=hp['dilation_exponential'],
        conv_channels=hp['conv_channels'], residual_channels=hp['residual_channels'],
        skip_channels=hp['skip_channels'], end_channels=hp['end_channels'],
        seq_length=seq_in_len, in_dim=1, out_dim=seq_out_len,
        layers=hp['layers'], propalpha=hp['propalpha'], 
        tanhalpha=hp['tanhalpha'], layer_norm_affline=False
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Forecasting from {forecast_start_date} for {seq_out_len} months")

    # Bayesian estimation
    num_runs = 10

    # Create a list to store the outputs
    outputs = []

    X = X.to(device)

    # Use model to predict next time step
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(X)  
            y_pred = output[-1, :, :,-1].clone()#36x142
        outputs.append(y_pred)

    # Stack the outputs along a new dimension
    outputs = torch.stack(outputs)

    Y=torch.mean(outputs,dim=0)
    variance = torch.var(outputs, dim=0)#variance
    std_dev = torch.std(outputs, dim=0)#standard deviation
    # Calculate 95% confidence interval
    z=1.96
    confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

    dat*=scale
    Y*=scale
    variance*=scale
    confidence*=scale

    print('output shape:',Y.shape)


    #Plotting:

    #save the data to desk
    dat=torch.from_numpy(dat)
    save_data(dat,Y,confidence,variance,col, forecast_start_date)

    #combine data
    all=torch.cat((dat,Y), dim=0)

    #scale down full data (global normalisation)
    incident_max=-999999999
    mention_max=-999999999

    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
            if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16,32):
                continue
            if 'Mention' in col[j]:
                if all[i,j]>mention_max:
                    mention_max=all[i,j]
            else:
                if all[i,j]>incident_max:
                    incident_max=all[i,j]

    all_n=torch.zeros(all.shape[0],all.shape[1])
    confidence_n=torch.zeros(confidence.shape[0],confidence.shape[1])
    u=0
    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
                if 'Mention' in col[j]:
                    all_n[i,j]=all[i,j]/mention_max
                else:
                    all_n[i,j]=all[i,j]/incident_max
                
                if i>=all.shape[0]-36:
                    confidence_n[u,j]=confidence[u,j]*(all_n[i,j]/all[i,j])
        if i>=all.shape[0]-36:
            u+=1

    #smoothing
    smoothed_dat=torch.stack(exponential_smoothing(all_n, 0.1))
    smoothed_confidence=torch.stack(exponential_smoothing(confidence_n, 0.1))

    #plot all forecasted nodes in the graph as groups of plots. Each plot consists of a single attack and its pertinent technologies
    for attack, solutions in graph.items():
        plot_forecast(smoothed_dat[:-36,],smoothed_dat[-36:,], smoothed_confidence,attack,solutions,index,col, forecast_start_year=forecast_start_year)
        save_gap(smoothed_dat[-36:,],attack,solutions,index, forecast_start_year) #save gaps of each attack to file

    # Save forecasts for external use
    forecasts_save_path = 'Dissertation/Bayesian MTGNN/model/forecasts_2025_2027.npy'
    np.save(forecasts_save_path, Y.cpu().numpy())
    print(f"Forecasts saved to {forecasts_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate forecasts using the trained model')
    parser.add_argument('--forecast_start_date', type=str, default='2025-01-01',
                       help='Start date for forecasting (format: YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(" B-MTGNN FORECAST GENERATION ")
    print(f"Generating 36-month forecasts starting from {args.forecast_start_date}")
    
    main(args.forecast_start_date)
    
    print("\n FORECASTING COMPLETED ")
    print("36-month forecasts: Dissertation/Bayesian MTGNN/model/forecasts_2025_2027.npy")