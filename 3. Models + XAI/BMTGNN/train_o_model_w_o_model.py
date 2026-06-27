import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from collections import defaultdict
import csv

from hyperparameter_optim_bmtgnn_transfer_learning import DataLoaderS, gtnet

def load_hyperparameters(hp_file='Dissertation/Bayesian MTGNN/model/hp.txt'):
    """Load the optimal hyperparameters from file"""
    with open(hp_file, 'r') as f:
        hp_str = f.read().strip()
    # Convert string representation to list
    hp_list = eval(hp_str)
    
    # Map to named parameters
    hyperparams = {
        'gcn_depth': hp_list[0],
        'lr': hp_list[1],
        'conv_channels': hp_list[2],
        'residual_channels': hp_list[3],
        'skip_channels': hp_list[4],
        'end_channels': hp_list[5],
        'subgraph_size': hp_list[6],
        'dropout': hp_list[7],
        'dilation_exponential': hp_list[8],
        'node_dim': hp_list[9],
        'propalpha': hp_list[10],
        'tanhalpha': hp_list[11],
        'layers': hp_list[12]
    }
    
    return hyperparams

def load_and_continue_training(model_path):
    """Load a pre-trained model and continue training on full data"""
    
    # Load optimal hyperparameters
    print("Loading optimal hyperparameters...")
    hp = load_hyperparameters()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file = 'Dissertation/Bayesian MTGNN/data/sm_data_g.csv'
    save_path = 'Dissertation/Bayesian MTGNN/model/o_model.pt'
    
    # Model configuration
    gcn_true = True
    buildA_true = True
    seq_in_len = 10
    seq_out_len = 36
    horizon = 1
    in_dim = 1
    normalize = 2
    # num_nodes will be determined from the DataLoader (Data.m) to avoid size mismatches
    num_nodes = None
    
    # Training configuration
    batch_size = 8
    epochs = 200  # Train for more epochs on full data
    clip = 10
    optim_method = 'adam'
    
    print("Loading full dataset...")
    # Use all data for training (train=1.0, valid=0.0)
    Data = DataLoaderS(data_file, 1.0, 0.0, device, horizon, seq_in_len, normalize, seq_out_len)
    
    # Derive number of nodes from the data loader to ensure adjacency and model sizes match
    num_nodes = getattr(Data, 'm', None)
    if num_nodes is None:
        raise RuntimeError("Unable to determine number of nodes from DataLoaderS (Data.m is missing).")
    
    # Ensure adjacency is moved to the correct device and has matching shape
    predefined_adj = getattr(Data, 'adj', None)
    if predefined_adj is None:
        predefined_adj = None
    else:
        predefined_adj = predefined_adj.to(device)
    
    print("Initializing model with optimal hyperparameters...")
    model = gtnet(gcn_true, buildA_true, hp['gcn_depth'], num_nodes,
                 device, predefined_adj, dropout=hp['dropout'], subgraph_size=hp['subgraph_size'],
                 node_dim=hp['node_dim'], dilation_exponential=hp['dilation_exponential'],
                 conv_channels=hp['conv_channels'], residual_channels=hp['residual_channels'],
                 skip_channels=hp['skip_channels'], end_channels=hp['end_channels'],
                 seq_length=seq_in_len, in_dim=in_dim, out_dim=seq_out_len,
                 layers=hp['layers'], propalpha=hp['propalpha'], 
                 tanhalpha=hp['tanhalpha'], layer_norm_affline=False)
    
    # Load pre-trained model weights
    print(f"Loading pre-trained model from {model_path}...")
    # Use safe loader to avoid size mismatch exceptions; copy only matching params
    transferred_keys, skipped_keys = safe_load_pretrained(model, model_path, device)
    print(f"Transferred {len(transferred_keys)} parameters from checkpoint. {len(skipped_keys)} parameters were skipped.")
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print(f"Model parameters: {nParams}")
    
    # Loss and optimizer
    criterion = nn.L1Loss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=0.00001)
    
    print("Continuing training on full dataset...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        n_samples = 0
        iter = 0
        
        for X, Y in Data.get_batches(Data.train[0], Data.train[1], batch_size, True):
            model.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            
            # Use all nodes (no splitting for final training)
            output = model(X)
            output = torch.squeeze(output, 3)
            scale = Data.scale.expand(output.size(0), output.size(1), Data.m)
            output *= scale
            Y *= scale
            
            loss = criterion(output, Y)
            loss.backward()
            
            # Gradient clipping
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * Data.m)
            iter += 1
        
        avg_loss = total_loss / n_samples
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Average Loss: {avg_loss:.4f}')
    
    print("Training completed. Saving final model...")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    return model

def safe_load_pretrained(model, checkpoint_path, device):
    """
    Load pretrained weights into model but only copy parameters that match in name and shape.
    This avoids size-mismatch errors when checkpoint architecture differs from the current model.
    Returns (transferred_keys, skipped_keys).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support common checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt_state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt_state = ckpt['state_dict']
    else:
        ckpt_state = ckpt

    model_state = model.state_dict()
    transferred = []
    skipped = []

    for name, param in model_state.items():
        if name in ckpt_state:
            ckpt_param = ckpt_state[name]
            # Ensure ckpt_param is a tensor
            if not isinstance(ckpt_param, torch.Tensor):
                try:
                    ckpt_param = torch.tensor(ckpt_param)
                except Exception:
                    skipped.append((name, 'not_tensor'))
                    continue

            if ckpt_param.shape == param.shape:
                try:
                    param.copy_(ckpt_param.to(param.device))
                    transferred.append(name)
                except Exception as e:
                    skipped.append((name, f'copy_error:{e}'))
            else:
                skipped.append((name, f'shape_mismatch_ckpt{tuple(ckpt_param.shape)}_model{tuple(param.shape)}'))
        else:
            skipped.append((name, 'missing_in_checkpoint'))

    # Report summary
    print(f"Pretrained loading summary: transferred {len(transferred)} params, skipped {len(skipped)} params")
    if skipped:
        # Show a few skipped entries for debugging
        for s in skipped[:20]:
            print(f"  SKIPPED: {s[0]} -> {s[1]}")
    return transferred, skipped

if __name__ == "__main__":
    print(" B-MTGNN CONTINUED MODEL TRAINING ")
    print("Loading pre-trained model and continuing training on full dataset (July 2011 - December 2024)")
    
    # Specify the path to pre-trained model
    model_path = 'Dissertation/Bayesian MTGNN/temp_best_model.pt'  # Update this path
    
    # Load and continue training the pre-trained model
    final_model = load_and_continue_training(model_path)
    
    print("\n PROCESS COMPLETED ")
    print("Final operational model: model/o_model.pt")
    print("36-month forecasts: model/forecasts_2025_2027.npy")