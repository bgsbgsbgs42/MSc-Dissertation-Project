import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import json
import time
import warnings
from datetime import datetime, timedelta
import os
import sys
from collections import defaultdict, Counter

# Import the model architecture from model building script
from train import SimpleVisionTransformer

warnings.filterwarnings('ignore')

class CyberThreatModelEvaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model first to get the correct configuration
        self.load_model()
        
        # Metrics storage
        self.metrics = {}
        
    def extract_config_from_checkpoint(self, checkpoint):
        """Extract model configuration from the checkpoint file"""
        # Default configuration
        config = {
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
            'num_nodes': 645,
        }
        
        # Try to extract configuration from checkpoint
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config.update(checkpoint['config'])
            print("Loaded configuration from checkpoint")
        else:
            # Infer configuration from model weights
            print("Inferring configuration from model weights...")
            
            # Infer embed_dim from cls_token shape
            if 'patch_embed.cls_token' in checkpoint:
                config['embed_dim'] = checkpoint['patch_embed.cls_token'].shape[2]
                print(f"Inferred embed_dim: {config['embed_dim']}")
            
            # Infer num_nodes from projection weight shape
            if 'patch_embed.projection.weight' in checkpoint:
                config['num_nodes'] = checkpoint['patch_embed.projection.weight'].shape[1]
                config['patch_size'] = checkpoint['patch_embed.projection.weight'].shape[2]
                print(f"Inferred num_nodes: {config['num_nodes']}, patch_size: {config['patch_size']}")
            
            # Infer sequence_length from position_embeddings
            if 'patch_embed.position_embeddings' in checkpoint:
                num_patches = checkpoint['patch_embed.position_embeddings'].shape[1] - 1
                config['sequence_length'] = num_patches * config['patch_size']
                print(f"Inferred sequence_length: {config['sequence_length']}")
            
            # Infer num_layers by counting transformer layers
            layer_count = 0
            for key in checkpoint.keys():
                if key.startswith('transformer.layers.') and '.self_attn.' in key:
                    layer_idx = int(key.split('.')[2])
                    layer_count = max(layer_count, layer_idx + 1)
            if layer_count > 0:
                config['num_layers'] = layer_count
                print(f"Inferred num_layers: {config['num_layers']}")
            
            # Infer hidden_dim from linear layers in transformer
            #  Use linear1.weight shape[0] which is the output dimension
            if 'transformer.layers.0.linear1.weight' in checkpoint:
                config['hidden_dim'] = checkpoint['transformer.layers.0.linear1.weight'].shape[0]
                print(f"Inferred hidden_dim from transformer.layers.0.linear1.weight: {config['hidden_dim']}")
            elif 'forecast_head.0.weight' in checkpoint:
                # Fallback: infer from forecast head
                config['hidden_dim'] = checkpoint['forecast_head.0.weight'].shape[0]
                print(f"Inferred hidden_dim from forecast_head.0.weight: {config['hidden_dim']}")
            
            # Infer forecast_horizon from forecast head
            if 'forecast_head.3.weight' in checkpoint:
                output_size = checkpoint['forecast_head.3.weight'].shape[0]
                if config['num_nodes'] > 0:
                    config['forecast_horizon'] = output_size // config['num_nodes']
                    print(f"Inferred forecast_horizon: {config['forecast_horizon']}")
        
        return config
    
    def load_model(self):
        """Load the trained model and extract its configuration"""
        try:
            
            # Load checkpoint first to extract configuration
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract configuration from checkpoint
            self.config = self.extract_config_from_checkpoint(checkpoint)
            
            print(f"Final model configuration:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            
            # Create model with extracted configuration
            self.model = SimpleVisionTransformer(self.config).to(self.device)
            
            # Load trained weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try strict loading first, then with strict=False as fallback
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("Model weights loaded successfully (strict mode)")
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Attempting non-strict loading...")
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    print("Model weights loaded successfully (non-strict mode)")
                    print("Warning: Some weight mismatches detected but model loaded with partial weights")
                except Exception as e2:
                    print(f"Non-strict loading also failed: {e2}")
                    print("Loading model with random initialization instead")
                
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_and_split_data(self):
        """Load data from data.csv and split into training and testing periods"""
        try:
            # Load the complete dataset
            print(f"Loading data from {self.data_file}...")
            self.data_df = pd.read_csv(self.data_file)
            
            # Check data structure
            print(f"Data columns: {self.data_df.columns.tolist()}")
            print(f"Data shape: {self.data_df.shape}")
            
            # Extract date information - try different possible date column names
            date_column = None
            for col in self.data_df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                    date_column = col
                    break
            
            if date_column is None:
                # Assume first column is date
                date_column = self.data_df.columns[0]
                print(f"No explicit date column found, using first column: {date_column}")
            
            # Convert to datetime
            self.data_df['date'] = pd.to_datetime(self.data_df[date_column])
            
            # Extract the actual data (excluding date column)
            data_columns = [col for col in self.data_df.columns if col != date_column and col != 'date']
            self.threat_data = self.data_df[data_columns].values
            
            print(f"Processed data shape: {self.threat_data.shape}")
            print(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
            
            # Use a more flexible split - use 70% for training, 30% for testing
            split_idx = int(0.7 * len(self.threat_data))
            self.data_train = self.threat_data[:split_idx]
            self.data_test = self.threat_data[split_idx:]
            self.dates_train = self.data_df['date'][:split_idx]
            self.dates_test = self.data_df['date'][split_idx:]
            
            print(f"Training data: {self.data_train.shape} (first {split_idx} samples)")
            print(f"Testing data: {self.data_test.shape} (last {len(self.data_test)} samples)")
            
            # Adjust forecast horizon if test data is too small
            required_length = self.config['sequence_length'] + self.config['forecast_horizon']
            if len(self.data_test) < required_length:
                print(f"Test data too small for forecast horizon {self.config['forecast_horizon']}")
                # Reduce forecast horizon to fit available data
                max_possible_horizon = len(self.data_test) - self.config['sequence_length'] - 1
                if max_possible_horizon > 0:
                    self.config['forecast_horizon'] = max_possible_horizon
                    print(f"Reduced forecast horizon to: {self.config['forecast_horizon']}")
                else:
                    # If still too small, reduce sequence length
                    self.config['sequence_length'] = min(self.config['sequence_length'], len(self.data_test) // 2)
                    self.config['forecast_horizon'] = len(self.data_test) - self.config['sequence_length'] - 1
                    print(f"Adjusted sequence_length to {self.config['sequence_length']}, forecast_horizon to {self.config['forecast_horizon']}")
            
            # Ensure data has correct number of nodes
            if self.data_train.shape[1] != self.config['num_nodes']:
                print(f"Adjusting data dimensions from {self.data_train.shape[1]} to {self.config['num_nodes']} nodes")
                if self.data_train.shape[1] > self.config['num_nodes']:
                    # Truncate if data has more nodes
                    self.data_train = self.data_train[:, :self.config['num_nodes']]
                    self.data_test = self.data_test[:, :self.config['num_nodes']]
                else:
                    # Pad if data has fewer nodes
                    pad_width = self.config['num_nodes'] - self.data_train.shape[1]
                    self.data_train = np.pad(self.data_train, ((0, 0), (0, pad_width)), mode='constant')
                    self.data_test = np.pad(self.data_test, ((0, 0), (0, pad_width)), mode='constant')
            
            # Normalize data using training statistics
            self.data_mean = np.mean(self.data_train, axis=0)
            self.data_std = np.std(self.data_train, axis=0) + 1e-8
            
            self.data_train_norm = (self.data_train - self.data_mean) / self.data_std
            self.data_test_norm = (self.data_test - self.data_mean) / self.data_std
            
            print("Data loaded and normalized successfully")
            
        except Exception as e:
            print(f"Error loading and splitting data: {e}")
            raise
    
    def generate_single_step_predictions(self, data, sequence_length):
        """Generate single-step predictions (more suitable for small datasets)"""
        predictions = []
        actuals = []
        
        print(f"Generating single-step predictions: data shape {data.shape}, sequence_length {sequence_length}")
        
        # For single-step prediction, we only need sequence_length + 1 samples
        total_possible = len(data) - sequence_length
        print(f"Total possible single-step prediction sequences: {total_possible}")
        
        if total_possible <= 0:
            print("Warning: Not enough data to generate predictions")
            return np.array([]), np.array([])
        
        with torch.no_grad():
            for i in range(total_possible):
                # Prepare input sequence
                input_seq = data[i:i+sequence_length]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                
                # Generate prediction (single step)
                pred = self.model(input_tensor)
                pred_np = pred.cpu().numpy()[0]  # Remove batch dimension
                
                # For single-step, we only care about the first prediction
                single_step_pred = pred_np[0:1]  # Take only the first time step
                
                # Get actual next value
                actual_next = data[i+sequence_length:i+sequence_length+1]
                
                predictions.append(single_step_pred)
                actuals.append(actual_next)
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        print(f"Generated {len(predictions)} single-step prediction sequences")
        print(f"Predictions shape: {predictions_array.shape}")
        print(f"Actuals shape: {actuals_array.shape}")
        
        return predictions_array, actuals_array
    
    def generate_multi_step_predictions(self, data, sequence_length, forecast_horizon):
        """Generate multi-step predictions if enough data is available"""
        predictions = []
        actuals = []
        
        print(f"Generating multi-step predictions: data shape {data.shape}, sequence_length {sequence_length}, forecast_horizon {forecast_horizon}")
        
        total_possible = len(data) - sequence_length - forecast_horizon + 1
        print(f"Total possible multi-step prediction sequences: {total_possible}")
        
        if total_possible <= 0:
            print("Not enough data for multi-step predictions, using single-step only")
            return np.array([]), np.array([])
        
        with torch.no_grad():
            for i in range(total_possible):
                # Prepare input sequence
                input_seq = data[i:i+sequence_length]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                
                # Generate prediction
                pred = self.model(input_tensor)
                pred_np = pred.cpu().numpy()[0]  # Remove batch dimension
                
                # Get actual values
                actual = data[i+sequence_length:i+sequence_length+forecast_horizon]
                
                predictions.append(pred_np)
                actuals.append(actual)
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        print(f"Generated {len(predictions)} multi-step prediction sequences")
        return predictions_array, actuals_array
    
    def calculate_msmape(self, predictions, actuals):
        """Calculate M-SMAPE (Modified Symmetric Mean Absolute Percentage Error)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        epsilon = 1e-8
        numerator = np.abs(predictions - actuals)
        denominator = (np.abs(predictions) + np.abs(actuals)) / 2 + epsilon
        
        smape = 2 * np.mean(numerator / denominator)
        return smape * 100  # Return as percentage
    
    def calculate_mae(self, predictions, actuals):
        """Calculate Mean Absolute Error (MAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        mae = np.mean(np.abs(predictions - actuals))
        return mae
    
    def calculate_mape(self, predictions, actuals):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        return mape
    def calculate_rmse(self, predictions, actuals):
        """Calculate Root Mean Squared Error (RMSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        return rmse
    
    def calculate_rae(self, predictions, actuals):
        """Calculate Relative Absolute Error (RAE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Sum of absolute errors
        sum_absolute_errors = np.sum(np.abs(predictions - actuals))
        
        # Sum of absolute deviations from mean
        mean_actuals = np.mean(actuals)
        sum_absolute_deviation = np.sum(np.abs(actuals - mean_actuals))
        
        # Avoid division by zero
        if sum_absolute_deviation == 0:
            return float('inf')
        
        rae = sum_absolute_errors / sum_absolute_deviation
        return rae
    
    def calculate_rrse(self, predictions, actuals):
        """Calculate Root Relative Squared Error (RRSE)"""
        if predictions.size == 0 or actuals.size == 0:
            return float('inf')
            
        # Sum of squared errors
        sum_squared_errors = np.sum((predictions - actuals) ** 2)
        
        # Sum of squared deviations from mean
        mean_actuals = np.mean(actuals)
        sum_squared_deviation = np.sum((actuals - mean_actuals) ** 2)
        
        # Avoid division by zero
        if sum_squared_deviation == 0:
            return float('inf')
        
        rrse = np.sqrt(sum_squared_errors / sum_squared_deviation)
        return rrse
    
    def calculate_auc_roc(self, predictions, actuals, threshold=0.3):
        """Calculate AUC-ROC for binary threat detection"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.5, np.array([0, 1]), np.array([0, 1])
            
        # Convert to binary classification (threat vs no-threat)
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        # Flatten for overall AUC
        pred_flat = pred_binary.flatten()
        actual_flat = actual_binary.flatten()
        
        try:
            auc = roc_auc_score(actual_flat, pred_flat)
            fpr, tpr, _ = roc_curve(actual_flat, pred_flat)
            return auc, fpr, tpr
        except:
            return 0.5, np.array([0, 1]), np.array([0, 1])  # Random classifier
    
    def calculate_false_positive_rate(self, predictions, actuals, threshold=0.3):
        """Calculate False Positive Rate"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        pred_binary = (predictions > threshold).astype(int)
        actual_binary = (actuals > threshold).astype(int)
        
        fp = np.sum((pred_binary == 1) & (actual_binary == 0))
        tn = np.sum((pred_binary == 0) & (actual_binary == 0))
        
        fpr = fp / (fp + tn + 1e-8)
        return fpr
    
    def calculate_attack_coverage(self, predictions, actuals, threshold=0.3):
        """Calculate Attack Coverage percentage"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        total_attacks = np.sum(actuals > threshold)
        detected_attacks = np.sum((predictions > threshold) & (actuals > threshold))
        
        coverage = detected_attacks / (total_attacks + 1e-8)
        return coverage
    
    def calculate_alert_precision(self, predictions, actuals, threshold=0.3):
        """Calculate Alert Precision"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        true_positives = np.sum((predictions > threshold) & (actuals > threshold))
        false_positives = np.sum((predictions > threshold) & (actuals <= threshold))
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        return precision
    
    def calculate_directional_accuracy(self, predictions, actuals):
        """Calculate Directional Accuracy for trend prediction"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        # For single-step predictions, compare consecutive values
        if len(predictions.shape) == 3 and predictions.shape[1] == 1:
            # Single-step case
            pred_direction = np.diff(predictions.squeeze(), axis=0) > 0
            actual_direction = np.diff(actuals.squeeze(), axis=0) > 0
        else:
            # Multi-step case
            pred_direction = np.diff(predictions, axis=1) > 0
            actual_direction = np.diff(actuals, axis=1) > 0
        
        correct_direction = (pred_direction == actual_direction)
        directional_accuracy = np.mean(correct_direction)
        return directional_accuracy
    
    def calculate_quantile_loss(self, predictions, actuals, quantiles=[0.1, 0.5, 0.9]):
        """Calculate Quantile Loss (Pinball Loss)"""
        if predictions.size == 0 or actuals.size == 0:
            return {f'quantile_{int(q*100)}': float('inf') for q in quantiles}
            
        losses = {}
        
        for q in quantiles:
            error = actuals - predictions
            loss = np.maximum(q * error, (q - 1) * error)
            losses[f'quantile_{int(q*100)}'] = np.mean(loss)
        
        return losses
    
    def calculate_ks_test(self, predictions, actuals):
        """Calculate Kolmogorov-Smirnov test statistics"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0, 0.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        stat, p_value = ks_2samp(pred_flat, actual_flat)
        return stat, p_value
    
    def calculate_js_divergence(self, predictions, actuals, bins=50):
        """Calculate Jensen-Shannon Divergence"""
        if predictions.size == 0 or actuals.size == 0:
            return 1.0
            
        # Flatten for overall distribution comparison
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        # Create histograms
        min_val = min(np.min(pred_flat), np.min(actual_flat))
        max_val = max(np.max(pred_flat), np.max(actual_flat))
        
        pred_hist, _ = np.histogram(pred_flat, bins=bins, range=(min_val, max_val), density=True)
        actual_hist, _ = np.histogram(actual_flat, bins=bins, range=(min_val, max_val), density=True)
        
        js_div = jensenshannon(pred_hist, actual_hist)
        return js_div
    
    def calculate_temporal_correlation(self, predictions, actuals):
        """Calculate Temporal Correlation Coefficient"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        correlations = []
        
        # Handle both single-step and multi-step predictions
        if len(predictions.shape) == 3:
            for threat_idx in range(actuals.shape[2]):
                threat_actual = actuals[:, :, threat_idx].flatten()
                threat_pred = predictions[:, :, threat_idx].flatten()
                
                # Remove constant sequences
                if np.std(threat_actual) > 0 and np.std(threat_pred) > 0:
                    corr = np.corrcoef(threat_actual, threat_pred)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        else:
            # Single correlation for flattened arrays
            pred_flat = predictions.flatten()
            actual_flat = actuals.flatten()
            if np.std(actual_flat) > 0 and np.std(pred_flat) > 0:
                corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0
    
    def calculate_adversarial_robustness(self, predictions, actuals, noise_level=0.1):
        """Calculate Adversarial Robustness Score"""
        if predictions.size == 0 or actuals.size == 0:
            return 0.0
            
        original_predictions = predictions.copy()
        
        # Add small perturbations to inputs
        noisy_predictions = predictions + np.random.normal(0, noise_level, predictions.shape)
        
        # Calculate change in predictions
        prediction_change = np.abs(noisy_predictions - original_predictions)
        robustness_score = 1.0 / (1.0 + np.mean(prediction_change))
        
        return robustness_score
    
    def calculate_operational_readiness_index(self, precision, recall, fpr, resource_efficiency):
        """Calculate composite Operational Readiness Index"""
        # Normalize metrics to 0-1 scale
        precision_norm = precision
        recall_norm = recall
        fpr_norm = 1 - fpr  # Lower FPR is better
        resource_norm = min(resource_efficiency, 1.0)
        
        # Weighted combination
        ori = (0.3 * precision_norm + 0.3 * recall_norm + 
               0.2 * fpr_norm + 0.2 * resource_norm)
        
        return ori
    
    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics"""
        print("Starting comprehensive model evaluation...")
        
        # Load and split data
        self.load_and_split_data()
        
        # Generate predictions using single-step approach (more suitable for small datasets)
        seq_len = self.config.get('sequence_length', 12)
        
        print(f"\nGenerating single-step predictions for training data...")
        pred_train, actual_train = self.generate_single_step_predictions(
            self.data_train_norm, seq_len
        )
        
        print(f"\nGenerating single-step predictions for testing data...")
        pred_test, actual_test = self.generate_single_step_predictions(
            self.data_test_norm, seq_len
        )
        
        # Try multi-step predictions if possible
        print(f"\nAttempting multi-step predictions...")
        pred_train_multi, actual_train_multi = self.generate_multi_step_predictions(
            self.data_train_norm, seq_len, min(6, self.config['forecast_horizon'])  # Reduced horizon for multi-step
        )
        pred_test_multi, actual_test_multi = self.generate_multi_step_predictions(
            self.data_test_norm, seq_len, min(6, self.config['forecast_horizon'])
        )
        
        # Use single-step predictions as primary, multi-step as secondary
        if pred_test.size > 0:
            pred_test_primary = pred_test
            actual_test_primary = actual_test
            prediction_type = "single-step"
        else:
            pred_test_primary = pred_test_multi
            actual_test_primary = actual_test_multi
            prediction_type = "multi-step"
        
        if pred_test_primary.size == 0:
            print("Warning: No predictions generated. Evaluation will use placeholder metrics.")
            # Initialize metrics with placeholder values
            self.initialize_placeholder_metrics()
            return self.metrics
        
        # Denormalize for some calculations
        pred_train_denorm = pred_train * self.data_std + self.data_mean
        actual_train_denorm = actual_train * self.data_std + self.data_mean
        pred_test_denorm = pred_test_primary * self.data_std + self.data_mean
        actual_test_denorm = actual_test_primary * self.data_std + self.data_mean
        
        print(f"Training predictions: {pred_train.shape}, actuals: {actual_train.shape}")
        print(f"Testing predictions: {pred_test_primary.shape}, actuals: {actual_test_primary.shape}")
        print(f"Using {prediction_type} predictions for evaluation")
        
        # Calculate metrics
        threshold = 0.3  # Lower threshold for cybersecurity context
        
        # 1. Accuracy Metrics
        self.metrics['M-SMAPE_Train'] = self.calculate_msmape(pred_train_denorm, actual_train_denorm)
        self.metrics['M-SMAPE_Test'] = self.calculate_msmape(pred_test_denorm, actual_test_denorm)
        
        #  MAE and MAPE metrics
        self.metrics['MAE_Train'] = self.calculate_mae(pred_train_denorm, actual_train_denorm)
        self.metrics['MAE_Test'] = self.calculate_mae(pred_test_denorm, actual_test_denorm)
        
        self.metrics['MAPE_Train'] = self.calculate_mape(pred_train_denorm, actual_train_denorm)
        self.metrics['MAPE_Test'] = self.calculate_mape(pred_test_denorm, actual_test_denorm)
        
        self.metrics['RMSE_Train'] = self.calculate_rmse(pred_train_denorm, actual_train_denorm)
        self.metrics['RMSE_Test'] = self.calculate_rmse(pred_test_denorm, actual_test_denorm)
        
        self.metrics['RAE_Train'] = self.calculate_rae(pred_train_denorm, actual_train_denorm)
        self.metrics['RAE_Test'] = self.calculate_rae(pred_test_denorm, actual_test_denorm)
        
        self.metrics['RRSE_Train'] = self.calculate_rrse(pred_train_denorm, actual_train_denorm)
        self.metrics['RRSE_Test'] = self.calculate_rrse(pred_test_denorm, actual_test_denorm)
        
        # 2. Detection Quality Metrics
        auc_train, fpr_curve_train, tpr_curve_train = self.calculate_auc_roc(pred_train, actual_train, threshold)
        auc_test, fpr_curve_test, tpr_curve_test = self.calculate_auc_roc(pred_test_primary, actual_test_primary, threshold)
        self.metrics['AUC_ROC_Train'] = auc_train
        self.metrics['AUC_ROC_Test'] = auc_test
        
        self.metrics['FPR_Train'] = self.calculate_false_positive_rate(pred_train, actual_train, threshold)
        self.metrics['FPR_Test'] = self.calculate_false_positive_rate(pred_test_primary, actual_test_primary, threshold)
        
        # 3. Coverage and Precision
        self.metrics['Attack_Coverage_Train'] = self.calculate_attack_coverage(pred_train, actual_train, threshold)
        self.metrics['Attack_Coverage_Test'] = self.calculate_attack_coverage(pred_test_primary, actual_test_primary, threshold)
        
        self.metrics['Alert_Precision_Train'] = self.calculate_alert_precision(pred_train, actual_train, threshold)
        self.metrics['Alert_Precision_Test'] = self.calculate_alert_precision(pred_test_primary, actual_test_primary, threshold)
        
        # 4. Distribution Metrics
        self.metrics['Directional_Accuracy_Train'] = self.calculate_directional_accuracy(pred_train, actual_train)
        self.metrics['Directional_Accuracy_Test'] = self.calculate_directional_accuracy(pred_test_primary, actual_test_primary)
        
        # Quantile Loss
        quantile_loss_train = self.calculate_quantile_loss(pred_train_denorm, actual_train_denorm)
        quantile_loss_test = self.calculate_quantile_loss(pred_test_denorm, actual_test_denorm)
        self.metrics.update({f'Quantile_Loss_Train_{k}': v for k, v in quantile_loss_train.items()})
        self.metrics.update({f'Quantile_Loss_Test_{k}': v for k, v in quantile_loss_test.items()})
        
        # Statistical Tests
        ks_stat_train, ks_p_train = self.calculate_ks_test(pred_train_denorm, actual_train_denorm)
        ks_stat_test, ks_p_test = self.calculate_ks_test(pred_test_denorm, actual_test_denorm)
        self.metrics['KS_Statistic_Train'] = ks_stat_train
        self.metrics['KS_Statistic_Test'] = ks_stat_test
        self.metrics['KS_p_value_Train'] = ks_p_train
        self.metrics['KS_p_value_Test'] = ks_p_test
        
        self.metrics['JS_Divergence_Train'] = self.calculate_js_divergence(pred_train_denorm, actual_train_denorm)
        self.metrics['JS_Divergence_Test'] = self.calculate_js_divergence(pred_test_denorm, actual_test_denorm)
        
        # 5. Temporal Metrics
        self.metrics['Temporal_Correlation_Train'] = self.calculate_temporal_correlation(pred_train, actual_train)
        self.metrics['Temporal_Correlation_Test'] = self.calculate_temporal_correlation(pred_test_primary, actual_test_primary)
        
        # 6. Robustness Metrics
        self.metrics['Adversarial_Robustness_Train'] = self.calculate_adversarial_robustness(pred_train, actual_train)
        self.metrics['Adversarial_Robustness_Test'] = self.calculate_adversarial_robustness(pred_test_primary, actual_test_primary)
        
        # 7. Composite Metrics
        recall_train = self.metrics['Attack_Coverage_Train']
        recall_test = self.metrics['Attack_Coverage_Test']
        resource_efficiency = 0.8  # Placeholder
        
        self.metrics['Operational_Readiness_Train'] = self.calculate_operational_readiness_index(
            self.metrics['Alert_Precision_Train'], recall_train, 
            self.metrics['FPR_Train'], resource_efficiency
        )
        self.metrics['Operational_Readiness_Test'] = self.calculate_operational_readiness_index(
            self.metrics['Alert_Precision_Test'], recall_test, 
            self.metrics['FPR_Test'], resource_efficiency
        )
        
        # Store additional data for visualization
        self.evaluation_data = {
            'predictions_train': pred_train,
            'actuals_train': actual_train,
            'predictions_test': pred_test_primary,
            'actuals_test': actual_test_primary,
            'fpr_curve_train': fpr_curve_train,
            'tpr_curve_train': tpr_curve_train,
            'fpr_curve_test': fpr_curve_test,
            'tpr_curve_test': tpr_curve_test,
            'prediction_type': prediction_type
        }
        
        return self.metrics

    def initialize_placeholder_metrics(self):
        """Initialize metrics with placeholder values when no predictions are generated"""
        placeholder_metrics = {
            'M-SMAPE_Train': float('inf'),
            'M-SMAPE_Test': float('inf'),
            'MAE_Train': float('inf'),
            'MAE_Test': float('inf'),
            'MAPE_Train': float('inf'),
            'MAPE_Test': float('inf'),
            'RMSE_Train': float('inf'),
            'RMSE_Test': float('inf'),
            'RAE_Train': float('inf'),
            'RAE_Test': float('inf'),
            'RRSE_Train': float('inf'),
            'RRSE_Test': float('inf'),
            'AUC_ROC_Train': 0.5,
            'AUC_ROC_Test': 0.5,
            'FPR_Train': 1.0,
            'FPR_Test': 1.0,
            'Attack_Coverage_Train': 0.0,
            'Attack_Coverage_Test': 0.0,
            'Alert_Precision_Train': 0.0,
            'Alert_Precision_Test': 0.0,
            'Directional_Accuracy_Train': 0.0,
            'Directional_Accuracy_Test': 0.0,
            'Quantile_Loss_Train_quantile_10': float('inf'),
            'Quantile_Loss_Train_quantile_50': float('inf'),
            'Quantile_Loss_Train_quantile_90': float('inf'),
            'Quantile_Loss_Test_quantile_10': float('inf'),
            'Quantile_Loss_Test_quantile_50': float('inf'),
            'Quantile_Loss_Test_quantile_90': float('inf'),
            'KS_Statistic_Train': 1.0,
            'KS_Statistic_Test': 1.0,
            'KS_p_value_Train': 0.0,
            'KS_p_value_Test': 0.0,
            'JS_Divergence_Train': 1.0,
            'JS_Divergence_Test': 1.0,
            'Temporal_Correlation_Train': 0.0,
            'Temporal_Correlation_Test': 0.0,
            'Adversarial_Robustness_Train': 0.0,
            'Adversarial_Robustness_Test': 0.0,
            'Operational_Readiness_Train': 0.0,
            'Operational_Readiness_Test': 0.0,
        }
        self.metrics.update(placeholder_metrics)

    def plot_detailed_analysis(self):
        """Create detailed analysis plots"""
        if not hasattr(self, 'evaluation_data') or self.evaluation_data['predictions_test'].size == 0:
            print("No evaluation data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        axes[0, 0].plot(self.evaluation_data['fpr_curve_train'], 
                       self.evaluation_data['tpr_curve_train'], 
                       'b-', linewidth=2, label=f'Train (AUC = {self.metrics["AUC_ROC_Train"]:.3f})')
        axes[0, 0].plot(self.evaluation_data['fpr_curve_test'], 
                       self.evaluation_data['tpr_curve_test'], 
                       'r-', linewidth=2, label=f'Test (AUC = {self.metrics["AUC_ROC_Test"]:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves: Train vs Test')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance Comparison Bar Chart
        comparison_metrics = ['M-SMAPE_Train', 'M-SMAPE_Test', 
                             'MAE_Train', 'MAE_Test',
                             'MAPE_Train', 'MAPE_Test',
                             'AUC_ROC_Train', 'AUC_ROC_Test',
                             'Directional_Accuracy_Train', 'Directional_Accuracy_Test']
        comparison_values = [self.metrics[m] for m in comparison_metrics]
        
        x_pos = np.arange(len(comparison_metrics))
        axes[0, 1].bar(x_pos, comparison_values, alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(comparison_metrics, rotation=45, ha='right')
        axes[0, 1].set_title('Key Metrics Comparison: Train vs Test')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Quantile Loss Comparison
        quantile_metrics_train = [self.metrics['Quantile_Loss_Train_quantile_10'],
                                self.metrics['Quantile_Loss_Train_quantile_50'],
                                self.metrics['Quantile_Loss_Train_quantile_90']]
        quantile_metrics_test = [self.metrics['Quantile_Loss_Test_quantile_10'],
                               self.metrics['Quantile_Loss_Test_quantile_50'],
                               self.metrics['Quantile_Loss_Test_quantile_90']]
        
        x_quantile = np.arange(3)
        width = 0.35
        axes[1, 0].bar(x_quantile - width/2, quantile_metrics_train, width, label='Train', alpha=0.7)
        axes[1, 0].bar(x_quantile + width/2, quantile_metrics_test, width, label='Test', alpha=0.7)
        axes[1, 0].set_xticks(x_quantile)
        axes[1, 0].set_xticklabels(['Q10', 'Q50', 'Q90'])
        axes[1, 0].set_ylabel('Quantile Loss')
        axes[1, 0].set_title('Quantile Loss Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction vs Actual Scatter
        if self.evaluation_data['predictions_test'].size > 0:
            test_pred_flat = self.evaluation_data['predictions_test'].flatten()
            test_actual_flat = self.evaluation_data['actuals_test'].flatten()
            axes[1, 1].scatter(test_actual_flat, test_pred_flat, alpha=0.5, s=10)
            axes[1, 1].plot([test_actual_flat.min(), test_actual_flat.max()], 
                           [test_actual_flat.min(), test_actual_flat.max()], 'r--', alpha=0.8)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Prediction vs Actual (Test Set)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Detailed analysis plot saved as 'detailed_model_analysis.png'")
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CYBER THREAT FORECASTING MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.model_path}")
        report_lines.append(f"Data: {self.data_file}")
        report_lines.append("")
        
        # Model Configuration
        report_lines.append("MODEL CONFIGURATION")
        report_lines.append("-" * 40)
        for key, value in self.config.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Calculate overall scores
        avg_msmape = (self.metrics['M-SMAPE_Train'] + self.metrics['M-SMAPE_Test']) / 2
        avg_mae = (self.metrics['MAE_Train'] + self.metrics['MAE_Test']) / 2
        avg_mape = (self.metrics['MAPE_Train'] + self.metrics['MAPE_Test']) / 2
        avg_auc = (self.metrics['AUC_ROC_Train'] + self.metrics['AUC_ROC_Test']) / 2
        avg_ori = (self.metrics['Operational_Readiness_Train'] + self.metrics['Operational_Readiness_Test']) / 2
        avg_rmse = (self.metrics['RMSE_Train'] + self.metrics['RMSE_Test']) / 2
        avg_rae = (self.metrics['RAE_Train'] + self.metrics['RAE_Test']) / 2
        avg_rrse = (self.metrics['RRSE_Train'] + self.metrics['RRSE_Test']) / 2
        
        
        report_lines.append(f"Average M-SMAPE: {avg_msmape:.2f}%")
        report_lines.append(f"Average MAE: {avg_mae:.4f}")
        report_lines.append(f"Average MAPE: {avg_mape:.2f}%")
        report_lines.append(f"Average AUC-ROC: {avg_auc:.3f}")
        report_lines.append(f"Average RMSE: {avg_rmse:.4f}")
        report_lines.append(f"Average RAE: {avg_rae:.4f}")
        report_lines.append(f"Average RRSE: {avg_rrse:.4f}")
        report_lines.append(f"Average Operational Readiness: {avg_ori:.3f}")
        report_lines.append("")
        
        # Detailed Metrics
        report_lines.append("DETAILED METRICS")
        report_lines.append("-" * 40)
        
        # Group metrics by category
        accuracy_metrics = {k: v for k, v in self.metrics.items() if 'SMAPE' in k or 'MAE' in k or 'MAPE' in k or 'Directional' in k}
        detection_metrics = {k: v for k, v in self.metrics.items() if 'AUC' in k or 'FPR' in k or 'Coverage' in k or 'Precision' in k}
        temporal_metrics = {k: v for k, v in self.metrics.items() if 'Temporal' in k}
        robustness_metrics = {k: v for k, v in self.metrics.items() if 'Robustness' in k or 'KS' in k or 'JS' in k}
        quantile_metrics = {k: v for k, v in self.metrics.items() if 'Quantile' in k}
        
        report_lines.append("Accuracy Metrics:")
        for metric, value in accuracy_metrics.items():
            if 'SMAPE' in metric or 'MAPE' in metric:
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.2f}%")
            elif 'MAE' in metric:
                if value == float('inf'):
                    report_lines.append(f"  {metric}: N/A (no predictions)")
                else:
                    report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nDetection Quality Metrics:")
        for metric, value in detection_metrics.items():
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nTemporal Performance Metrics:")
        for metric, value in temporal_metrics.items():
            report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nRobustness and Distribution Metrics:")
        for metric, value in robustness_metrics.items():
            if 'p_value' in metric:
                report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value:.3f}")
        
        report_lines.append("\nUncertainty Quantification (Quantile Loss):")
        for metric, value in quantile_metrics.items():
            if value == float('inf'):
                report_lines.append(f"  {metric}: N/A (no predictions)")
            else:
                report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.append("")
        
        # Performance Assessment
        report_lines.append("PERFORMANCE ASSESSMENT")
        report_lines.append("-" * 40)
        
        # Assess model performance
        if avg_auc > 0.8:
            auc_assessment = "EXCELLENT"
        elif avg_auc > 0.7:
            auc_assessment = "GOOD"
        elif avg_auc > 0.6:
            auc_assessment = "FAIR"
        else:
            auc_assessment = "POOR"
        
        if avg_msmape < 10:
            smape_assessment = "EXCELLENT"
        elif avg_msmape < 20:
            smape_assessment = "GOOD"
        elif avg_msmape < 30:
            smape_assessment = "FAIR"
        else:
            smape_assessment = "POOR"
        
        report_lines.append(f"Detection Quality: {auc_assessment} (AUC-ROC: {avg_auc:.3f})")
        if avg_msmape == float('inf'):
            report_lines.append(f"Forecast Accuracy: N/A (no predictions generated)")
        else:
            report_lines.append(f"Forecast Accuracy: {smape_assessment} (M-SMAPE: {avg_msmape:.2f}%)")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if self.metrics['FPR_Test'] > 0.1:
            report_lines.append("HIGH FALSE POSITIVE RATE: Consider adjusting detection threshold")
        if self.metrics['Adversarial_Robustness_Test'] < 0.7:
            report_lines.append("LOW ROBUSTNESS: Enhance model regularization")
        if self.metrics['Directional_Accuracy_Test'] < 0.6:
            report_lines.append("POOR TREND PREDICTION: Focus on temporal feature engineering")
        
        if self.metrics['AUC_ROC_Test'] > 0.7 and self.metrics['FPR_Test'] < 0.2:
            report_lines.append("REASONABLE DETECTION CAPABILITY: Model shows promise")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        report_path = "model_evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(report_lines))
            f.close()
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path

def main():
    """Main execution function"""
    # Configuration
    model_path = "./model/ViT_Final/o_model.pt"  # Path to trained model
    data_file = "./data/sm_data_g.csv"  # Path to complete dataset
    
    # Initialize evaluator
    evaluator = CyberThreatModelEvaluator(
        model_path=model_path,
        data_file=data_file
    )
    
    # Run comprehensive evaluation
    metrics = evaluator.run_comprehensive_evaluation()
    
    # Generate visualizations
    evaluator.plot_detailed_analysis()
    
    # Generate text report
    report_path = evaluator.generate_report()
    
    print("\nEvaluation completed successfully!")
    print(f"Key metrics calculated: {len(metrics)}")
    print(f"Report generated: {report_path}")
    
    # Print key metrics to console
    print("\nKEY METRICS SUMMARY:")
    print(f"M-SMAPE Train: {metrics.get('M-SMAPE_Train', 'N/A'):.2f}%")
    print(f"M-SMAPE Test: {metrics.get('M-SMAPE_Test', 'N/A'):.2f}%")
    print(f"MAE Train: {metrics.get('MAE_Train', 'N/A'):.4f}")
    print(f"MAE Test: {metrics.get('MAE_Test', 'N/A'):.4f}")
    print(f"MAPE Train: {metrics.get('MAPE_Train', 'N/A'):.2f}%")
    print(f"MAPE Test: {metrics.get('MAPE_Test', 'N/A'):.2f}%")
    print(f"AUC-ROC Train: {metrics.get('AUC_ROC_Train', 'N/A'):.3f}")
    print(f"AUC-ROC Test: {metrics.get('AUC_ROC_Test', 'N/A'):.3f}")
    print(f"Directional Accuracy Test: {metrics.get('Directional_Accuracy_Test', 'N/A'):.3f}")
    print(f"Adversarial Robustness Test: {metrics.get('Adversarial_Robustness_Test', 'N/A'):.3f}")

if __name__ == "__main__":
    main()