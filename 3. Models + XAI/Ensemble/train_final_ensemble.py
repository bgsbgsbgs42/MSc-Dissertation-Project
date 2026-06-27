import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
import argparse

from hyperparameter_optimization_ensemble_pretraining import (
    SpatioTemporalEnsemble, DataLoaderEnsemble, 
    NumpyEncoder, set_random_seed
)

def create_directories():
    directories = ['model/Ensemble']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_hyperparameters():
    try:
        with open('Dissertation/Ensemble Variant/model/Ensemble/hp.txt', 'r') as f:
            hp = json.load(f)
        print("Loaded optimized hyperparameters from hp.txt")
        return hp
    except FileNotFoundError:
        print("Hyperparameter file not found. Using default hyperparameters.")
        return get_default_hyperparameters()

def get_default_hyperparameters():
    return {
        'sequence_length': 12,
        'vit_patch_size': 4,
        'vit_embed_dim': 128,
        'vit_num_heads': 8,
        'vit_hidden_dim': 256,
        'vit_num_layers': 3,
        'graph_hidden_dim': 128,
        'graph_num_layers': 2,
        'fusion_dim': 256,
        'dropout': 0.1,
        'mc_dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 8,
        'forecast_horizon': 36,
        'weight_decay': 1e-5
    }

def prepare_full_dataset(data_file, sequence_length, forecast_horizon, device):
    print(f"Loading full dataset from: {data_file}")
    
    data = pd.read_csv(data_file)
    raw_data = data.values
    feature_names = list(data.columns)
    
    print(f"Full dataset shape: {raw_data.shape}")
    print(f"Number of nodes: {raw_data.shape[1]}")
    
    # Normalize the data
    scale = np.max(np.abs(raw_data), axis=0)
    scale[scale == 0] = 1.0
    normalized_data = raw_data / scale
    
    # Create sequences from the entire dataset
    n_sequences = len(normalized_data) - sequence_length - forecast_horizon + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough data to create sequences. Need {sequence_length + forecast_horizon} samples, but have {len(normalized_data)}")
    
    X = np.zeros((n_sequences, sequence_length, raw_data.shape[1]))
    Y = np.zeros((n_sequences, forecast_horizon, raw_data.shape[1]))
    
    for i in range(n_sequences):
        X[i] = normalized_data[i:i+sequence_length]
        Y[i] = normalized_data[i+sequence_length:i+sequence_length+forecast_horizon]
    
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float()
    
    print(f"Created {n_sequences} sequences for full training")
    
    return X_tensor, Y_tensor, scale, feature_names

def train_final_model(data_file, device, epochs=100):
    set_random_seed(123)
    create_directories()
    
    # Load hyperparameters
    hp = load_hyperparameters()
    
    # Prepare full dataset
    X_full, Y_full, scale, feature_names = prepare_full_dataset(
        data_file, 
        hp['sequence_length'], 
        hp['forecast_horizon'],
        device
    )
    
    hp['num_nodes'] = X_full.shape[2]
    
    print(f"Final model configuration:")
    for key, value in hp.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    model = SpatioTemporalEnsemble(hp).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=hp['learning_rate'], 
        weight_decay=hp.get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting final training on full dataset...")
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Simple batching (shuffle indices)
        indices = torch.randperm(X_full.size(0))
        batch_size = min(hp['batch_size'], X_full.size(0))
        
        for i in range(0, X_full.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_full[batch_indices].to(device)
            Y_batch = Y_full[batch_indices].to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            
            # Ensure output and target have same dimensions
            min_horizon = min(output.shape[1], Y_batch.shape[1])
            output = output[:, :min_horizon, :]
            Y_batch = Y_batch[:, :min_horizon, :]
            
            # Denormalize for loss calculation
            scale_tensor = torch.from_numpy(scale).float().to(device)
            scale_batch = scale_tensor.expand(output.size(0), output.size(1), output.size(2))
            
            output_denorm = output * scale_batch
            Y_denorm = Y_batch * scale_batch
            
            loss = criterion(output_denorm, Y_denorm)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save operational model
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': hp,
                'scale': scale,
                'feature_names': feature_names,
                'final_loss': avg_loss,
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat()
            }, 'Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt')
            
            print(f"  -> Saved new best model with loss: {avg_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 20:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for final save
    checkpoint = torch.load('Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save training history
    training_history = {
        'hyperparameters': hp,
        'train_losses': train_losses,
        'best_loss': best_loss,
        'final_epoch': len(train_losses),
        'dataset_size': X_full.size(0),
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('model/Ensemble/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n FINAL TRAINING COMPLETE ")
    print(f"Best training loss: {best_loss:.6f}")
    print(f"Final model saved as: model/Ensemble/o_model.pt")
    print(f"Training history saved as: model/Ensemble/training_history.json")
    print(f"Model can forecast {hp['forecast_horizon']} months from the end of 2024")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Final Ensemble Model')
    parser.add_argument('--data', type=str, default='Dissertation/Ensemble Variant/data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(" FINAL ENSEMBLE MODEL TRAINING ")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Epochs: {args.epochs}")
    
    try:
        model = train_final_model(args.data, device, args.epochs)
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()