import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd

# Import the classes from the hyperparameter optimization script
from hyperoptim_transfer_learning_rework import (
    set_random_seed, NumpyEncoder, CyberThreatDataset, 
    VisionTransformerForTimeSeries, DataLoaderS
)

def safe_load_pretrained(model, ckpt_state, device):
    """
    Copy matching parameters from ckpt_state (a dict of tensors) into model.
    Try:
      1) exact shape copy
      2) partial copy of overlapping slices when shapes differ but ranks match
    Returns lists of transferred and skipped parameter names for debugging.
    """
    model_state = model.state_dict()
    transferred = []
    skipped = []

    for name, param in model_state.items():
        if name not in ckpt_state:
            skipped.append((name, 'missing_in_checkpoint'))
            continue

        ckpt_param = ckpt_state[name]
        # Ensure ckpt_param is a tensor
        if not isinstance(ckpt_param, torch.Tensor):
            try:
                ckpt_param = torch.tensor(ckpt_param)
            except Exception:
                skipped.append((name, 'not_tensor'))
                continue

        # Exact shape match -> copy directly
        if tuple(ckpt_param.shape) == tuple(param.shape):
            try:
                param.copy_(ckpt_param.to(param.device))
                transferred.append(name)
                continue
            except Exception as e:
                skipped.append((name, f'copy_error:{e}'))
                continue

        # Attempt partial copy when ranks match and both are at least 1-D
        if ckpt_param.ndim == param.ndim and ckpt_param.ndim >= 1:
            try:
                # Build slices for overlapping region along each dim
                slices_ckpt = tuple(slice(0, min(s_ck, s_mod)) for s_ck, s_mod in zip(ckpt_param.shape, param.shape))
                slices_mod  = slices_ckpt  # same start:0..min
                # Create views and copy overlapping block
                param_view = param.__getitem__(slices_mod)
                ckpt_view = ckpt_param.__getitem__(slices_ckpt).to(param.device)
                # Ensure same dtype
                if ckpt_view.dtype != param_view.dtype:
                    ckpt_view = ckpt_view.to(dtype=param_view.dtype)
                param_view.copy_(ckpt_view)
                transferred.append(name + '_partial')
                continue
            except Exception as e:
                skipped.append((name, f'partial_copy_error:{e}'))
                continue

        # Otherwise skip
        skipped.append((name, f'shape_mismatch_ckpt{tuple(ckpt_param.shape)}_model{tuple(param.shape)}'))

    # Report summary
    print(f"[safe_load_pretrained] transferred {len(transferred)} params, skipped {len(skipped)} params")
    if skipped:
        for s in skipped[:30]:
            print(f"  SKIPPED: {s[0]} -> {s[1]}")
    return transferred, skipped

def main():
    parser = argparse.ArgumentParser(description='ViT Final Model Training')
    parser.add_argument('--data', type=str, default='Dissertation/Vision Transformer/data/sm_data_g.csv', help='location of the data file')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--model_path', type=str, default='Dissertation/Vision Transformer/final_transfer_model/final_model.pt', help='path to pre-trained model to continue training')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Set fixed random seed
    set_random_seed(123)
    
    # Create output directory
    os.makedirs('model/ViT', exist_ok=True)
    
    # Load best hyperparameters
    try:
        with open('Dissertation/Vision Transformer/model/ViT/hp.txt', 'r') as f:
            best_hp = json.load(f)
        print("Loaded best hyperparameters from hp.txt")
    except FileNotFoundError:
        print("hp.txt not found. Using default hyperparameters.")
        best_hp = {
            'sequence_length': 12,
            'patch_size': 4,
            'embed_dim': 128,
            'num_heads': 8,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'mc_dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 8,
            'forecast_horizon': 36,
            'num_nodes': 142  # Will be updated from data
        }
    
    # Load full data for final training
    print("Loading full dataset for final training...")
    # Read into DataFrame and keep only numeric columns (coerce non-numeric)
    df = pd.read_csv(args.data)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        # Try coercing all columns to numeric (non-numeric -> NaN)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs produced by coercion: forward/backward fill then fill remaining with zeros
    numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

    full_data = numeric_df.values.astype(float)

    # Update num_nodes based on actual numeric data
    best_hp['num_nodes'] = full_data.shape[1]

    # Compute mean/std safely and guard against zero std
    data_mean = np.nanmean(full_data, axis=0)
    data_std = np.nanstd(full_data, axis=0)
    data_std[data_std == 0] = 1e-8
    normalized_data = (full_data - data_mean) / (data_std + 1e-8)

    # Create dataset using entire data
    dataset = CyberThreatDataset(
        normalized_data, 
        sequence_length=best_hp['sequence_length'],
        forecast_horizon=best_hp['forecast_horizon']
    )

    # Create data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=best_hp['batch_size'],
        shuffle=True
    )

    print(f"Training on {len(dataset)} samples")
    print(f"Final hyperparameters (before checkpoint update): {best_hp}")

    # Initialize resume variables before inspecting checkpoint
    start_epoch = 1
    best_loss = float('inf')

    # Load checkpoint FIRST to update architecture params ----
    checkpoint = None
    if os.path.exists(args.model_path):
        print(f"Found checkpoint at {args.model_path}, loading to inspect config...")
        checkpoint = torch.load(args.model_path, map_location=device)
        # Normalize checkpoint state access
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ckpt_state = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            ckpt_state = checkpoint['state_dict']
        else:
            ckpt_state = checkpoint

        # If checkpoint contains config, update architecture-related hyperparameters
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            arch_params = ['sequence_length', 'patch_size', 'embed_dim', 'num_heads', 
                          'hidden_dim', 'num_layers', 'dropout', 'mc_dropout', 'num_nodes', 'forecast_horizon']
            for param in arch_params:
                if param in checkpoint_config:
                    best_hp[param] = checkpoint_config[param]
            print("Updated hyperparameters from checkpoint config.")

        # If checkpoint contains data_mean/std, prefer those for normalization
        if isinstance(checkpoint, dict):
            if 'data_mean' in checkpoint and 'data_std' in checkpoint:
                try:
                    data_mean = np.array(checkpoint['data_mean'], dtype=float)
                    data_std = np.array(checkpoint['data_std'], dtype=float)
                    print("Using data_mean/data_std from checkpoint for normalization.")
                except Exception:
                    pass

    else:
        print(f"No pre-trained model found at {args.model_path}. Starting fresh training.")

    print(f"Final hyperparameters (used to instantiate model): {best_hp}")

    # Initialize model (after applying checkpoint config updates)
    model = VisionTransformerForTimeSeries(best_hp).to(device)

    # Try to load weights: strict first, otherwise safe copy
    if checkpoint is not None:
        # Determine ckpt_state if not already set
        if 'ckpt_state' not in locals():
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                ckpt_state = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                ckpt_state = checkpoint['state_dict']
            else:
                ckpt_state = checkpoint

        try:
            # First try strict load
            model.load_state_dict(ckpt_state)
            print("Checkpoint loaded successfully (strict). All weights transferred.")
        except Exception as e:
            print(f"Strict load_state_dict failed: {e}")
            print("Falling back to safe_load_pretrained to copy matching parameters.")
            transferred, skipped = safe_load_pretrained(model, ckpt_state, device)
    else:
        print("No checkpoint to load; model initialized with random weights.")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_hp['learning_rate'], 
        weight_decay=0.00001
    )

    # Load optimizer state if available in checkpoint (after creating optimizer)
    if checkpoint is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    patience_counter = 0
    
    print("Starting final model training...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}')
        
        avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save operational model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': best_hp,
                'data_mean': data_mean,
                'data_std': data_std,
                'epoch': epoch,
                'loss': best_loss
            }, 'model/ViT/o_model.pt')
            
            print(f"Saved new best model with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= 20:
            print("Early stopping triggered")
            break
    
    print(f"Final training completed. Best loss: {best_loss:.6f}")
    
    # Save training configuration and results
    training_summary = {
        'hyperparameters': best_hp,
        'final_loss': best_loss,
        'total_epochs': epoch,
        'timestamp': datetime.now().isoformat(),
        'data_shape': full_data.shape,
        'device': str(device),
        'resumed_from': args.model_path if os.path.exists(args.model_path) else None
    }
    
    with open('model/ViT/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
    
    print("Operational model saved as 'model/ViT/o_model.pt'")
    print("Training summary saved as 'model/ViT/training_summary.json'")

if __name__ == "__main__":
    main()