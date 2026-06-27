import numpy as np
import os
import torch
import torch.nn as nn
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import shap
import lime
import lime.lime_tabular
import warnings
import re
warnings.filterwarnings('ignore')

plt.rcParams['savefig.dpi'] = 1200
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})


# DEFINE MODEL CLASSES

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

# HELPER FUNCTIONS
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

# XAI MODULE - EXPLAINABLE AI METHODS

class XAIExplainer:
    """Comprehensive XAI module for time series forecasting model."""
    
    def __init__(self, model: nn.Module, config: dict, feature_names: List[str]):
        """
        Initialize XAI explainer.
        
        Args:
            model: Trained VisionTransformerForTimeSeries model
            config: Model configuration dictionary
            feature_names: List of feature/node names
        """
        self.model = model
        self.config = config
        self.feature_names = feature_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"XAI Explainer initialized with {len(feature_names)} features")
        print(f"Using device: {self.device}")
    
    # SHAP EXPLANATIONS
    
    def compute_shap_values(self, data: np.ndarray, num_samples: int = 100) -> Dict:
        """
        Compute SHAP values for the model predictions.
        
        Args:
            data: Input data of shape (n_samples, seq_len, n_features)
            num_samples: Number of samples for SHAP approximation
            
        Returns:
            Dictionary containing SHAP values and related information
        """
        print("\n" + "="*60)
        print("COMPUTING SHAP VALUES")
        print("="*60)
        
        if data.ndim != 3:
            raise ValueError("SHAP expects data shaped (samples, seq_len, n_features)")
        _, seq_len, n_features = data.shape
        
        total_flat_features = seq_len * n_features
        max_flat_features = min(256, total_flat_features)
        if total_flat_features > max_flat_features:
            feature_indices = np.sort(np.random.choice(total_flat_features, max_flat_features, replace=False))
        else:
            feature_indices = np.arange(total_flat_features)
        baseline_flat = data.reshape(len(data), -1).mean(axis=0).astype(np.float32)
        feature_names_flat = [
            f"{self.feature_names[idx % n_features]}_t{idx // n_features}"
            for idx in feature_indices
        ]
        with torch.no_grad():
            sample_pred = self.model(torch.FloatTensor(data[:1]).to(self.device), mc_dropout=False)
        pred_dim = sample_pred.reshape(1, -1).shape[1]
        max_output_dim = min(32, pred_dim)
        
        # Create background dataset for SHAP (using random samples)
        background_size = min(20, len(data))
        background_indices = np.random.choice(len(data), background_size, replace=False)
        background_data = data[background_indices]
        background_full = background_data.reshape(background_size, -1).astype(np.float32)
        background_data_flat = background_full[:, feature_indices]
    
        # Define SHAP explainer function
        def model_predict(input_data: np.ndarray) -> np.ndarray:
            """Wrapper function for model prediction."""
            if len(input_data) == 0:
                return np.zeros((0, max_output_dim), dtype=np.float32)
            batch_flat = np.tile(baseline_flat, (len(input_data), 1))
            batch_flat[:, feature_indices] = input_data
            batch = batch_flat.reshape(-1, seq_len, n_features)
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            with torch.no_grad():
                pred = self.model(batch_tensor, mc_dropout=False).cpu().numpy().reshape(len(batch), -1)
            return pred[:, :max_output_dim].astype(np.float32)
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model_predict,
            background_data_flat,
            link="identity"
        )
        explainer.max_samples = int(min(64, max(16, seq_len * 2)))
        
        # Select test samples
        test_size = min(10, num_samples, len(data))
        test_indices = np.random.choice(len(data), test_size, replace=False)
        test_data = data[test_indices]
        test_full = test_data.reshape(test_size, -1).astype(np.float32)
        test_data_flat = test_full[:, feature_indices]
        
        # Compute SHAP values
        print(f"Computing SHAP values for {test_size} samples...")
        num_shap_samples = int(min(64, max(16, seq_len * 2)))
        shap_raw = explainer.shap_values(test_data_flat, nsamples=num_shap_samples)
        shap_values = self._reshape_shap_outputs(shap_raw, test_size, max_output_dim, len(feature_indices))
        
        selected_feature_labels = feature_names_flat[:shap_values.shape[-1]]
        
        print(f"SHAP values shape: {shap_values.shape}")
        
        # Aggregate SHAP values
        shap_summary = {
            'shap_values': shap_values,
            'test_data': test_data,
            'feature_names': self.feature_names,
            'feature_names_flat': selected_feature_labels,
            'global_importance': self._compute_global_shap_importance(shap_values, selected_feature_labels),
            'local_explanations': self._extract_local_explanations(shap_values, test_data, selected_feature_labels)
        }
        
        return shap_summary
    def _compute_global_shap_importance(self, shap_values: np.ndarray, feature_labels: List[str]) -> Dict:
        """Compute global feature importance from SHAP values."""
        abs_shap = np.abs(shap_values)
        avg_importance = np.mean(abs_shap, axis=(0, 1))
        aggregated: Dict[str, float] = defaultdict(float)
        used_labels = feature_labels[:len(avg_importance)]
        for label, value in zip(used_labels, avg_importance):
            base_feature = label.split('_t')[0]
            aggregated[base_feature] += float(value)
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    def _reshape_shap_outputs(self, shap_values: Any, num_samples: int, output_dim: int, feature_dim: int) -> np.ndarray:
        """Reshape Kernel SHAP outputs back to (samples, outputs, selected_features)."""
        if isinstance(shap_values, list):
            reshaped = []
            for sv in shap_values[:output_dim]:
                sv_arr = np.array(sv, dtype=np.float32)
                if sv_arr.ndim == 1:
                    sv_arr = np.tile(sv_arr, (num_samples, 1))
                elif sv_arr.ndim == 2 and sv_arr.shape[0] != num_samples and sv_arr.shape[1] == num_samples:
                    sv_arr = sv_arr.T
                sv_arr = sv_arr.reshape(num_samples, -1)[:, :feature_dim]
                reshaped.append(sv_arr)
            return np.stack(reshaped, axis=1)
        arr = np.array(shap_values, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.tile(arr, (num_samples, 1))
        if arr.ndim == 2:
            arr = arr.reshape(num_samples, 1, -1)
        elif arr.ndim == 3:
            if arr.shape[0] == output_dim:
                arr = arr.transpose(1, 0, 2)
            elif arr.shape[1] != output_dim:
                arr = arr.reshape(num_samples, -1, arr.shape[-1])
        if arr.shape[-1] > feature_dim:
            arr = arr[..., :feature_dim]
        if arr.shape[1] > output_dim:
            arr = arr[:, :output_dim, :]
        if arr.ndim != 3:
            raise ValueError("Unexpected SHAP output shape.")
        return arr
    
    def _extract_local_explanations(self, shap_values: np.ndarray, 
                                   test_data: np.ndarray,
                                   feature_labels: List[str]) -> List[Dict]:
        """Extract local explanations for each test sample."""
        local_explanations = []
        for i in range(min(5, len(test_data))):
            sample_shap = np.abs(shap_values[i]).mean(axis=0)
            aggregated: Dict[str, float] = defaultdict(float)
            used_labels = feature_labels[:sample_shap.shape[0]]
            for label, value in zip(used_labels, sample_shap):
                aggregated[label.split('_t')[0]] += float(value)
            top_items = dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:10])
            local_explanations.append({
                'sample_index': i,
                'top_features': top_items,
                'input_summary': {
                    'mean': float(np.mean(test_data[i])),
                    'std': float(np.std(test_data[i])),
                    'min': float(np.min(test_data[i])),
                    'max': float(np.max(test_data[i]))
                }
            })
        return local_explanations
    
    def plot_shap_summary(self, shap_summary: Dict, save_path: str = None):
        """Create comprehensive SHAP visualization."""
        shap_values = shap_summary['shap_values']
        test_data = shap_summary['test_data']
        feature_names_flat = shap_summary.get('feature_names_flat', [])
        
        # Create directory for plots
        if save_path is None:
            save_path = 'model/ViT/xai/shap/'
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Global feature importance bar plot
        plt.figure(figsize=(12, 8))
        global_importance = shap_summary['global_importance']
        features = list(global_importance.keys())[:20]  # Top 20 features
        importance_values = [global_importance[f] for f in features]
        
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)')
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'global_importance.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'global_importance.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        # 2. SHAP summary plot (beeswarm)
        try:
            if shap_values.ndim == 3:
                shap_values_flat = shap_values.mean(axis=1)
                used_labels = feature_names_flat[:shap_values_flat.shape[1]]
                test_data_flat = test_data.reshape(test_data.shape[0], -1)[:, :shap_values_flat.shape[1]]
                plt.figure(figsize=(14, 8))
                shap.summary_plot(
                    shap_values_flat,
                    test_data_flat,
                    feature_names=used_labels,
                    show=False,
                    max_display=20
                )
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'shap_summary.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(save_path, 'shap_summary.pdf'), bbox_inches='tight', format='pdf')
                plt.close()
        except Exception as e:
            print(f"Could not create SHAP summary plot: {e}")
        
        # 3. Local explanation for first sample
        if shap_summary['local_explanations']:
            local_exp = shap_summary['local_explanations'][0]
            
            plt.figure(figsize=(10, 6))
            features = list(local_exp['top_features'].keys())[:10]
            values = list(local_exp['top_features'].values())[:10]
            
            plt.barh(range(len(features)), values)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Contribution (|SHAP value|)')
            plt.title(f'Local Explanation for Sample {local_exp["sample_index"]}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'local_explanation.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, 'local_explanation.pdf'), bbox_inches='tight', format='pdf')
            plt.close()
        
        print(f"SHAP plots saved to: {save_path}")
    
    # ATTENTION VISUALIZATION
    
    def extract_attention_maps(self, input_data: np.ndarray) -> Dict:
        """
        Extract attention maps from the transformer model.
        
        Args:
            input_data: Input data of shape (batch_size, seq_len, n_features)
            
        Returns:
            Dictionary containing attention maps and related information
        """
        print("\n" + "="*60)
        print("EXTRACTING ATTENTION MAPS")
        print("="*60)
        
        # Hook to capture attention weights
        attention_maps = []
        
        def attention_hook(module, input, output):
            """Hook function to capture attention weights."""
            attn_tensor = None
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_tensor = output[1]
            if attn_tensor is None:
                return
            attention_maps.append(attn_tensor.detach().cpu().numpy())
        hooks = []
        for name, module in self.model.named_modules():
            if 'self_attn' in name or 'attention' in name:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass with hooks
        input_tensor = torch.FloatTensor(input_data[:1]).to(self.device)  # Use first sample
        with torch.no_grad():
            _ = self.model(input_tensor, mc_dropout=False)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if not attention_maps:
            print("Warning: No attention maps captured (attention weights not returned).")
            return {
                'attention_maps': {},
                'patch_info': self._extract_patch_information(input_data[0]),
                'input_sample': input_data[0],
                'num_layers': 0
            }
        
        # Process attention maps
        processed_maps = {}
        for i, attn in enumerate(attention_maps):
            # attn shape: (batch_size, num_heads, seq_len, seq_len)
            if len(attn.shape) == 4:
                # Average across batch and heads for visualization
                attn_avg = np.mean(attn, axis=(0, 1))
                processed_maps[f'layer_{i}'] = attn_avg
            else:
                processed_maps[f'layer_{i}'] = attn
        
        # Extract patch information
        patch_info = self._extract_patch_information(input_data[0])
        
        return {
            'attention_maps': processed_maps,
            'patch_info': patch_info,
            'input_sample': input_data[0],
            'num_layers': len(attention_maps)
        }
    
    def _extract_patch_information(self, sample: np.ndarray) -> Dict:
        """Extract information about patches for visualization."""
        seq_len = sample.shape[0]
        patch_size = self.config.get('patch_size', 1)
        num_patches = seq_len // patch_size
        
        patch_info = {
            'seq_len': seq_len,
            'patch_size': patch_size,
            'num_patches': num_patches,
            'patch_boundaries': [(i * patch_size, (i + 1) * patch_size) 
                                for i in range(num_patches)]
        }
        
        return patch_info
    
    def visualize_attention(self, attention_info: Dict, save_path: str = None):
        """Visualize attention maps from the transformer."""
        if not attention_info:
            print("No attention information to visualize")
            return
        
        # Create directory for plots
        if save_path is None:
            save_path = 'model/ViT/xai/attention/'
        os.makedirs(save_path, exist_ok=True)
        
        attention_maps = attention_info['attention_maps']
        patch_info = attention_info['patch_info']
        
        # 1. Attention matrix for each layer
        for layer_name, attn_matrix in attention_maps.items():
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            im = plt.imshow(attn_matrix, cmap='viridis', aspect='auto')
            
            # Add patch boundaries if available
            if patch_info['num_patches'] > 1:
                patch_boundaries = patch_info['patch_boundaries']
                for boundary in patch_boundaries:
                    plt.axvline(x=boundary[0] - 0.5, color='white', linestyle='--', alpha=0.5)
                    plt.axhline(y=boundary[0] - 0.5, color='white', linestyle='--', alpha=0.5)
            
            plt.colorbar(im, label='Attention Weight')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.title(f'Attention Matrix - {layer_name}')
            plt.tight_layout()
            
            # Save figure
            filename = f'attention_{layer_name}.png'
            plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, filename.replace('.png', '.pdf')), 
                       bbox_inches='tight', format='pdf')
            plt.close()
        
        # 2. Attention head diversity (if multi-head attention)
        self._visualize_attention_head_diversity(attention_maps, save_path)
        
        # 3. Temporal attention patterns
        self._visualize_temporal_attention(attention_info, save_path)
        
        print(f"Attention visualizations saved to: {save_path}")
    
    def _visualize_attention_head_diversity(self, attention_maps: Dict, save_path: str):
        """Visualize diversity across attention heads."""
        for layer_name, attn_matrix in attention_maps.items():
            # Check if we have multi-head attention data
            if len(attn_matrix.shape) == 3:  # (num_heads, seq_len, seq_len)
                num_heads = attn_matrix.shape[0]
                
                fig, axes = plt.subplots(1, min(4, num_heads), figsize=(16, 4))
                if num_heads == 1:
                    axes = [axes]
                
                for i in range(min(4, num_heads)):
                    ax = axes[i]
                    im = ax.imshow(attn_matrix[i], cmap='viridis', aspect='auto')
                    ax.set_title(f'Head {i}')
                    ax.set_xlabel('Key')
                    ax.set_ylabel('Query')
                    plt.colorbar(im, ax=ax)
                
                plt.suptitle(f'Attention Head Diversity - {layer_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'head_diversity_{layer_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _visualize_temporal_attention(self, attention_info: Dict, save_path: str):
        """Visualize attention patterns over time."""
        attention_maps = attention_info['attention_maps']
        sample = attention_info['input_sample']
        
        # Use first layer attention
        if attention_maps:
            first_layer = list(attention_maps.keys())[0]
            attn_matrix = attention_maps[first_layer]
            
            # Compute attention to CLS token (if exists)
            if attn_matrix.shape[0] == attn_matrix.shape[1]:
                # Assuming CLS token is at position 0
                cls_attention = attn_matrix[:, 0]  # Attention from all positions to CLS
                
                plt.figure(figsize=(12, 6))
                
                # Plot attention to CLS token over time
                plt.plot(cls_attention[1:], 'b-', linewidth=2, label='Attention to CLS token')
                plt.xlabel('Time Position (excluding CLS)')
                plt.ylabel('Attention Weight')
                plt.title('Temporal Attention Pattern (to CLS token)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add input signal for context
                ax2 = plt.gca().twinx()
                normalized_sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
                mean_signal = np.mean(normalized_sample, axis=1)
                ax2.plot(mean_signal, 'r--', alpha=0.5, linewidth=1, label='Input signal (mean)')
                ax2.set_ylabel('Normalized Input')
                ax2.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'temporal_attention.png'), 
                           dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(save_path, 'temporal_attention.pdf'), 
                           bbox_inches='tight', format='pdf')
                plt.close()
    
    # LIME EXPLANATIONS
    
    def compute_lime_explanations(self, data: np.ndarray, num_samples: int = 5) -> Dict:
        """
        Compute LIME explanations for model predictions.
        
        Args:
            data: Input data of shape (n_samples, seq_len, n_features)
            num_samples: Number of samples to explain
            
        Returns:
            Dictionary containing LIME explanations
        """
        print("\n" + "="*60)
        print("COMPUTING LIME EXPLANATIONS")
        print("="*60)
        
        # Flatten data for LIME (treat each time-feature combination as separate feature)
        n_samples, seq_len, n_features = data.shape
        data_flat = data.reshape(n_samples, seq_len * n_features)
        
        # Create feature names for flattened data
        feature_names_flat = []
        for t in range(seq_len):
            for f in range(n_features):
                feature_names_flat.append(f"{self.feature_names[f]}_t{t}")
        
        # Define prediction function
        def predict_proba(input_data: np.ndarray) -> np.ndarray:
            """Prediction function for LIME."""
            # Reshape back to original format
            input_reshaped = input_data.reshape(-1, seq_len, n_features)
            
            batch_size = 32
            predictions = []
            
            for i in range(0, len(input_reshaped), batch_size):
                batch = input_reshaped[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                with torch.no_grad():
                    pred = self.model(batch_tensor, mc_dropout=False)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            # For binary classification-like explanation, convert to probabilities
            # For regression, we can explain the predicted value directly
            return predictions.reshape(len(input_data), -1)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=data_flat,
            feature_names=feature_names_flat,
            mode='regression',
            verbose=True,
            random_state=42
        )
        
        # Generate explanations for each sample
        explanations = []
        for i in range(min(num_samples, len(data_flat))):
            print(f"Computing LIME explanation for sample {i+1}/{min(num_samples, len(data_flat))}")
            
            exp = explainer.explain_instance(
                data_row=data_flat[i],
                predict_fn=predict_proba,
                num_features=20,  # Top 20 features
                num_samples=1000  # Number of samples to generate
            )
            
            # Extract explanation details
            exp_dict = {
                'sample_index': i,
                'local_pred': predict_proba(data_flat[i:i+1])[0],
                'feature_importance': exp.as_list(),
                'intercept': exp.intercept,
                'score': exp.score
            }
            
            explanations.append(exp_dict)
        
        return {
            'explanations': explanations,
            'feature_names_flat': feature_names_flat,
            'original_shape': (seq_len, n_features)
        }
    
    def visualize_lime_explanations(self, lime_info: Dict, save_path: str = None):
        """Visualize LIME explanations."""
        if save_path is None:
            save_path = 'model/ViT/xai/lime/'
        os.makedirs(save_path, exist_ok=True)
        
        explanations = lime_info['explanations']
        feature_names_flat = lime_info['feature_names_flat']
        seq_len, n_features = lime_info['original_shape']
        
        for i, exp in enumerate(explanations):
            # Create figure
            plt.figure(figsize=(14, 8))
            
            # Parse feature importance
            features = []
            importance = []
            
            for feature, weight in exp['feature_importance'][:15]:  # Top 15 features
                features.append(feature)
                importance.append(weight)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            colors = ['red' if w < 0 else 'blue' for w in importance]
            
            plt.barh(y_pos, importance, color=colors)
            plt.yticks(y_pos, features)
            plt.xlabel('Feature Weight')
            plt.title(f'LIME Explanation for Sample {exp["sample_index"]}\n'
                     f'Local Prediction: {exp["local_pred"][:3]}... | Score: {exp["score"]:.3f}')
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'lime_sample_{i}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, f'lime_sample_{i}.pdf'), 
                       bbox_inches='tight', format='pdf')
            plt.close()
        
        # Create summary of important temporal patterns
        self._create_lime_temporal_summary(lime_info, save_path)
        
        print(f"LIME visualizations saved to: {save_path}")
    
    def _create_lime_temporal_summary(self, lime_info: Dict, save_path: str):
        """Create summary of temporal patterns from LIME explanations."""
        explanations = lime_info['explanations']
        feature_names_flat = lime_info['feature_names_flat']
        seq_len, n_features = lime_info['original_shape']
        
        # Aggregate feature importance across samples
        feature_importance_agg = defaultdict(float)
        
        for exp in explanations:
            for feature, weight in exp['feature_importance']:
                feature_importance_agg[feature] += abs(weight)
        
        # Parse temporal information from feature names
        temporal_importance = np.zeros(seq_len)
        feature_importance = np.zeros(n_features)
        
        for feature, total_importance in feature_importance_agg.items():
            parts = feature.split('_t', 1)
            if len(parts) == 2:
                feature_name = parts[0].strip()
                time_part = parts[1]
                match = re.search(r'\d+', time_part)
                if not match:
                    continue
                time_step = int(match.group())
                
                # Find feature index
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    feature_importance[feature_idx] += total_importance
                
                if 0 <= time_step < seq_len:
                    temporal_importance[time_step] += total_importance
        
        # Plot temporal importance
        plt.figure(figsize=(12, 6))
        plt.plot(temporal_importance, 'b-', linewidth=2, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Aggregated Feature Importance')
        plt.title('Temporal Importance Pattern (from LIME)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'lime_temporal_pattern.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        feature_names_short = [name[:20] + '...' if len(name) > 20 else name 
                              for name in self.feature_names]
        sorted_indices = np.argsort(feature_importance)[::-1][:20]
        
        plt.barh(range(len(sorted_indices)), 
                feature_importance[sorted_indices])
        plt.yticks(range(len(sorted_indices)), 
                  [feature_names_short[i] for i in sorted_indices])
        plt.xlabel('Aggregated Importance')
        plt.title('Top 20 Features by LIME Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'lime_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # BAYESIAN CAUSAL MODELING
    
    def analyze_causal_relationships(self, data: np.ndarray, 
                                    target_indices: List[int] = None) -> Dict:
        """
        Analyze causal relationships using Bayesian methods.
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            target_indices: Indices of target features to analyze
            
        Returns:
            Dictionary containing causal analysis results
        """
        print("\n" + "="*60)
        print("BAYESIAN CAUSAL ANALYSIS")
        print("="*60)
        
        if target_indices is None:
            # Analyze relationships for all features
            target_indices = list(range(min(10, data.shape[1])))
        
        # Prepare data for causal analysis
        data_df = pd.DataFrame(data, columns=self.feature_names)
        
        # Compute Granger causality (simplified version)
        causal_results = {}
        for target_idx in target_indices:
            if target_idx >= len(self.feature_names):
                continue
            
            target_name = self.feature_names[target_idx]
            print(f"Analyzing causal relationships for: {target_name}")
            
            # Simplified causal analysis using correlation and lagged relationships
            causal_strengths = self._compute_causal_strengths(data, target_idx)
            
            causal_results[target_name] = {
                'causes': causal_strengths,
                'target_index': target_idx,
                'target_mean': float(np.mean(data[:, target_idx])),
                'target_std': float(np.std(data[:, target_idx]))
            }
        
        # Perform structural causal model inference (simplified)
        scm_results = self._infer_structural_model(data, target_indices)
        
        return {
            'causal_results': causal_results,
            'structural_model': scm_results,
            'feature_names': self.feature_names
        }
    
    def _compute_causal_strengths(self, data: np.ndarray, target_idx: int, 
                                 max_lag: int = 3) -> Dict:
        """Compute causal strengths using cross-correlation with lags."""
        causal_strengths = {}
        
        for i in range(data.shape[1]):
            if i == target_idx:
                continue
            
            feature_name = self.feature_names[i]
            max_corr = 0
            best_lag = 0
            
            # Compute cross-correlation with different lags
            for lag in range(1, max_lag + 1):
                if lag < len(data):
                    # Align series with lag
                    series1 = data[lag:, target_idx]
                    series2 = data[:-lag, i]
                    
                    if len(series1) > 1 and len(series2) > 1:
                        corr = np.corrcoef(series1, series2)[0, 1]
                        if not np.isnan(corr) and abs(corr) > abs(max_corr):
                            max_corr = corr
                            best_lag = lag
            
            causal_strengths[feature_name] = {
                'correlation': float(max_corr),
                'lag': best_lag,
                'abs_strength': float(abs(max_corr))
            }
        
        # Sort by absolute strength
        sorted_strengths = dict(sorted(causal_strengths.items(),
                                      key=lambda x: x[1]['abs_strength'],
                                      reverse=True))
        
        return sorted_strengths
    
    def _infer_structural_model(self, data: np.ndarray, 
                               target_indices: List[int]) -> Dict:
        """Infer a simplified structural causal model."""
        n_features = data.shape[1]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        # Threshold for significant relationships
        threshold = 0.3
        
        # Create adjacency matrix
        adjacency = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    # Direction based on temporal precedence (simplified)
                    if i in target_indices or j in target_indices:
                        # For targets, use correlation sign and magnitude
                        adjacency[i, j] = corr_matrix[i, j]
        
        # Extract significant relationships
        relationships = []
        for i in range(n_features):
            for j in range(n_features):
                if abs(adjacency[i, j]) > threshold:
                    relationships.append({
                        'from': self.feature_names[i],
                        'to': self.feature_names[j],
                        'strength': float(adjacency[i, j]),
                        'type': 'positive' if adjacency[i, j] > 0 else 'negative'
                    })
        
        return {
            'adjacency_matrix': adjacency,
            'relationships': relationships,
            'correlation_matrix': corr_matrix,
            'threshold': threshold
        }
    
    def visualize_causal_analysis(self, causal_info: Dict, save_path: str = None):
        """Visualize causal analysis results."""
        if save_path is None:
            save_path = 'model/ViT/xai/causal/'
        os.makedirs(save_path, exist_ok=True)
        
        causal_results = causal_info['causal_results']
        structural_model = causal_info['structural_model']
        
        # 1. Causal strength plot for each target
        for target_name, result in causal_results.items():
            causes = result['causes']
            
            if not causes:
                continue
            
            plt.figure(figsize=(12, 8))
            
            # Get top 10 causes
            top_causes = list(causes.items())[:10]
            cause_names = [name for name, _ in top_causes]
            strengths = [info['abs_strength'] for _, info in top_causes]
            correlations = [info['correlation'] for _, info in top_causes]
            
            # Create horizontal bar plot with color indicating direction
            colors = ['red' if c < 0 else 'blue' for c in correlations]
            y_pos = np.arange(len(cause_names))
            
            plt.barh(y_pos, strengths, color=colors)
            plt.yticks(y_pos, cause_names)
            plt.xlabel('Causal Strength (|correlation|)')
            plt.title(f'Top Causal Influences on {target_name}\n'
                     f'Target stats: μ={result["target_mean"]:.3f}, σ={result["target_std"]:.3f}')
            
            # Add lag information
            for i, (name, info) in enumerate(top_causes):
                lag_str = f" (lag={info['lag']})" if info['lag'] > 0 else ""
                plt.text(strengths[i] + 0.01, i, 
                        f"{info['correlation']:.2f}{lag_str}",
                        va='center')
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'causal_{target_name.replace("/", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation matrix heatmap
        plt.figure(figsize=(14, 12))
        corr_matrix = structural_model['correlation_matrix']
        
        # Limit to top features if too many
        max_features = 30
        if len(self.feature_names) > max_features:
            # Select features with highest variance
            variances = np.var(corr_matrix, axis=0)
            top_indices = np.argsort(variances)[::-1][:max_features]
            corr_matrix = corr_matrix[top_indices][:, top_indices]
            feature_names_subset = [self.feature_names[i] for i in top_indices]
        else:
            feature_names_subset = self.feature_names
        
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names_subset,
                   yticklabels=feature_names_subset,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation'})
        
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'correlation_matrix.pdf'), 
                   bbox_inches='tight', format='pdf')
        plt.close()
        
        # 3. Causal network visualization
        self._visualize_causal_network(structural_model, save_path)
        
        print(f"Causal analysis visualizations saved to: {save_path}")
    
    def _visualize_causal_network(self, structural_model: Dict, save_path: str):
        """Visualize causal network using networkx."""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes
            for i, feature in enumerate(self.feature_names):
                G.add_node(feature)
            
            # Add edges with significant relationships
            threshold = structural_model['threshold']
            adjacency = structural_model['adjacency_matrix']
            
            for i in range(len(self.feature_names)):
                for j in range(len(self.feature_names)):
                    strength = adjacency[i, j]
                    if abs(strength) > threshold:
                        G.add_edge(self.feature_names[i], 
                                  self.feature_names[j],
                                  weight=abs(strength),
                                  sign='positive' if strength > 0 else 'negative')
            
            # Create visualization
            plt.figure(figsize=(16, 12))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Separate positive and negative edges
            pos_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d['sign'] == 'positive']
            neg_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d['sign'] == 'negative']
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=500, 
                                  node_color='lightblue', 
                                  alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
                                  width=2, alpha=0.7, 
                                  edge_color='green',
                                  connectionstyle="arc3,rad=0.1")
            
            nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                                  width=2, alpha=0.7, 
                                  edge_color='red', style='dashed',
                                  connectionstyle="arc3,rad=0.1")
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Add legend
            plt.plot([], [], color='green', linewidth=2, label='Positive influence')
            plt.plot([], [], color='red', linestyle='dashed', linewidth=2, 
                    label='Negative influence')
            plt.legend(loc='upper left')
            
            plt.title('Causal Network of Features\n'
                     f'(Threshold: |correlation| > {threshold})')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'causal_network.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, 'causal_network.pdf'), 
                       bbox_inches='tight', format='pdf')
            plt.close()
            
        except ImportError:
            print("NetworkX not installed. Skipping network visualization.")
        except Exception as e:
            print(f"Error creating network visualization: {e}")
    
    # DICE - COUNTERFACTUAL EXPLANATIONS
    
    def generate_counterfactuals(self, data: np.ndarray, 
                                target_indices: List[int],
                                desired_outcome: str = 'decrease',
                                num_counterfactuals: int = 5) -> Dict:
        """
        Generate counterfactual explanations using DiCE-like approach.
        
        Args:
            data: Input data of shape (n_samples, seq_len, n_features)
            target_indices: Indices of target features to modify
            desired_outcome: 'increase' or 'decrease' target values
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            Dictionary containing counterfactual explanations

        """
        print("\n" + "="*60)
        print("GENERATING COUNTERFACTUAL EXPLANATIONS")
        print("="*60)
        
        # Use first sample as reference
        reference_sample = data[0:1]
        
        # Get original prediction
        with torch.no_grad():
            reference_tensor = torch.FloatTensor(reference_sample).to(self.device)
            original_prediction = self.model(reference_tensor, mc_dropout=False)
            original_prediction = original_prediction.cpu().numpy()[0]
        
        counterfactuals = []
        
        for cf_idx in range(num_counterfactuals):
            # Generate counterfactual by modifying the input
            counterfactual = self._generate_single_counterfactual(
                reference_sample.copy(),
                target_indices,
                desired_outcome,
                cf_idx
            )
            
            # Get counterfactual prediction
            with torch.no_grad():
                cf_tensor = torch.FloatTensor(counterfactual).to(self.device)
                cf_prediction = self.model(cf_tensor, mc_dropout=False)
                cf_prediction = cf_prediction.cpu().numpy()[0]
            
            # Calculate changes
            input_changes = counterfactual[0] - reference_sample[0]
            prediction_changes = cf_prediction - original_prediction
            
            counterfactuals.append({
                'counterfactual_input': counterfactual[0],
                'counterfactual_prediction': cf_prediction,
                'input_changes': input_changes,
                'prediction_changes': prediction_changes,
                'original_input': reference_sample[0],
                'original_prediction': original_prediction,
                'modification_summary': self._summarize_modifications(
                    input_changes, 
                    target_indices,
                    self.feature_names
                )
            })
        
        return {
            'counterfactuals': counterfactuals,
            'target_indices': target_indices,
            'target_names': [self.feature_names[i] for i in target_indices],
            'desired_outcome': desired_outcome,
            'original_sample': reference_sample[0]
        }
    
    def _generate_single_counterfactual(self, sample: np.ndarray,
                                       target_indices: List[int],
                                       desired_outcome: str,
                                       cf_idx: int) -> np.ndarray:
        """Generate a single counterfactual example."""
        # Simple strategy: modify recent time steps more than distant ones
        seq_len = sample.shape[1]
        n_features = sample.shape[2]
        
        # Create modification mask (more modification to recent steps)
        time_weights = np.linspace(0.1, 1.0, seq_len)  # Recent steps weighted more
        
        for t in range(seq_len):
            for f_idx in target_indices:
                if f_idx < n_features:
                    # Apply modification based on desired outcome
                    if desired_outcome == 'increase':
                        modification = 0.5 * time_weights[t] * (cf_idx + 1)
                    else:  # decrease
                        modification = -0.5 * time_weights[t] * (cf_idx + 1)
                    
                    sample[0, t, f_idx] += modification
        
        return sample
    
    def _summarize_modifications(self, changes: np.ndarray,
                                target_indices: List[int],
                                feature_names: List[str]) -> List[Dict]:
        """Summarize the modifications made to generate counterfactual."""
        summary = []
        seq_len = changes.shape[0]
        
        for f_idx in target_indices:
            if f_idx >= len(feature_names):
                continue
            
            feature_name = feature_names[f_idx]
            
            # Calculate statistics of changes for this feature
            feature_changes = changes[:, f_idx]
            mean_change = np.mean(feature_changes)
            max_change = np.max(np.abs(feature_changes))
            num_modified = np.sum(np.abs(feature_changes) > 0.01)
            
            # Find time steps with largest changes
            if num_modified > 0:
                top_indices = np.argsort(np.abs(feature_changes))[-3:][::-1]
                top_changes = [(int(idx), float(feature_changes[idx])) 
                              for idx in top_indices]
            else:
                top_changes = []
            
            summary.append({
                'feature': feature_name,
                'mean_change': float(mean_change),
                'max_abs_change': float(max_change),
                'num_modified_timesteps': int(num_modified),
                'top_modifications': top_changes
            })
        
        return summary
    
    def visualize_counterfactuals(self, dice_info: Dict, save_path: str = None):
        """Visualize counterfactual explanations."""
        if save_path is None:
            save_path = 'model/ViT/xai/dice/'
        os.makedirs(save_path, exist_ok=True)
        
        counterfactuals = dice_info['counterfactuals']
        target_names = dice_info['target_names']
        original_sample = dice_info['original_sample']
        
        # 1. Compare original vs counterfactual predictions
        plt.figure(figsize=(14, 10))
        
        num_cfs = len(counterfactuals)
        #colors = plt.cm.viridis(np.linspace(0, 1, num_cfs))
        
        for i, cf in enumerate(counterfactuals):
            original_pred = cf['original_prediction']
            cf_pred = cf['counterfactual_prediction']
            
            # Plot for each target
            for j, target_name in enumerate(target_names):
                if j < original_pred.shape[1]:
                    plt.subplot(len(target_names), 1, j + 1)
                    
                    plt.plot(original_pred[:, j], 'k-', linewidth=3, 
                            label='Original' if i == 0 else None)
                    plt.plot(cf_pred[:, j], '-',  
                            linewidth=2, alpha=0.7,
                            label=f'CF {i+1}' if j == 0 else None)
                    
                    plt.title(f'Predictions for {target_name}')
                    plt.xlabel('Forecast Horizon')
                    plt.ylabel('Predicted Value')
                    plt.grid(True, alpha=0.3)
                    
                    if j == 0:
                        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'counterfactual_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'counterfactual_predictions.pdf'), 
                   bbox_inches='tight', format='pdf')
        plt.close()
        
        # 2. Input modifications for each counterfactual
        for i, cf in enumerate(counterfactuals):
            plt.figure(figsize=(16, 12))
            
            modifications = cf['modification_summary']
            
            for j, mod_summary in enumerate(modifications):
                plt.subplot(len(modifications), 1, j + 1)
                
                feature_name = mod_summary['feature']
                changes = cf['input_changes'][:, self.feature_names.index(feature_name)]
                
                # Plot original and modified values
                original_values = original_sample[:, self.feature_names.index(feature_name)]
                modified_values = cf['counterfactual_input'][:, self.feature_names.index(feature_name)];
                
                plt.plot(original_values, 'k-', linewidth=2, label='Original')
                plt.plot(modified_values, 'r--', linewidth=2, label='Modified')
                
                # Highlight changes
                change_indices = np.where(np.abs(changes) > 0.01)[0]
                if len(change_indices) > 0:
                    plt.scatter(change_indices, modified_values[change_indices],
                               color='blue', s=50, zorder=5, 
                               label='Modified points')
                
                plt.title(f'{feature_name} - Mean change: {mod_summary["mean_change"]:.3f}')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Counterfactual {i+1} - Input Modifications', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'counterfactual_{i}_modifications.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Summary of all counterfactuals
        self._create_counterfactual_summary(dice_info, save_path)
        
        print(f"Counterfactual visualizations saved to: {save_path}")
    
    def _create_counterfactual_summary(self, dice_info: Dict, save_path: str):
        """Create summary table of counterfactuals."""
        counterfactuals = dice_info['counterfactuals']
        target_names = dice_info['target_names']
        
        # Create summary table
        summary_data = []
        
        for i, cf in enumerate(counterfactuals):
            row = {'Counterfactual': i + 1}
            
            # Calculate overall change in predictions
            pred_change = cf['prediction_changes']
            
            for j, target in enumerate(target_names):
                if j < pred_change.shape[1]:
                    mean_change = np.mean(pred_change[:, j])
                    max_change = np.max(np.abs(pred_change[:, j]))
                    row[f'{target}_mean'] = float(mean_change)
                    row[f'{target}_max'] = float(max_change)
            
            # Count significant modifications
            input_changes = cf['input_changes']
            num_significant = np.sum(np.abs(input_changes) > 0.1)
            row['num_significant_mods'] = int(num_significant)
            
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(save_path, 'counterfactual_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Create visualization of summary
        plt.figure(figsize=(12, 8))
        
        # Plot mean changes for each counterfactual
        for j, target in enumerate(target_names):
            if f'{target}_mean' in summary_df.columns:
                plt.plot(summary_df['Counterfactual'], 
                        summary_df[f'{target}_mean'],
                        'o-', linewidth=2, markersize=8,
                        label=target)
        
        plt.xlabel('Counterfactual')
        plt.ylabel('Mean Prediction Change')
        plt.title('Counterfactual Effectiveness Summary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'counterfactual_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Counterfactual summary saved to: {summary_file}")
    
    
    # COMPREHENSIVE XAI REPORT
    
    
    def generate_comprehensive_report(self, data: np.ndarray, 
                                    report_dir: str = 'model/ViT/xai/report/') -> Dict:
        """
        Generate comprehensive XAI report with all methods.
        
        Args:
            data: Input data for analysis
            report_dir: Directory to save report
            
        Returns:
            Dictionary containing all XAI results
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE XAI REPORT")
        print("="*80)
        
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. SHAP Analysis
        
        print("\n1. Performing SHAP analysis...")
        shap_results = self.compute_shap_values(data[:100], num_samples=50)
        self.plot_shap_summary(shap_results, os.path.join(report_dir, 'shap/'))
        
        # 2. Attention Visualization
        print("\n2. Extracting attention maps...")
        attention_results = self.extract_attention_maps(data[:5])
        self.visualize_attention(attention_results, os.path.join(report_dir, 'attention/'))
        
        # 3. LIME Explanations
        print("\n3. Computing LIME explanations...")
        lime_results = self.compute_lime_explanations(data[:10], num_samples=5)
        self.visualize_lime_explanations(lime_results, os.path.join(report_dir, 'lime/'))
        
        
    
        # 4. Bayesian Causal Analysis
        print("\n4. Performing Bayesian causal analysis...")
        # Use first 1000 timesteps for causal analysis
        causal_data = data[:min(1000, len(data))]
        if len(causal_data.shape) == 3:
            causal_data = causal_data.reshape(-1, causal_data.shape[2])
        
        causal_results = self.analyze_causal_relationships(causal_data)
        self.visualize_causal_analysis(causal_results, os.path.join(report_dir, 'causal/'))
        
        # 5. DiCE Counterfactuals
        print("\n5. Generating counterfactual explanations...")
        # Select first 3 features as targets for counterfactuals
        target_indices = list(range(min(3, len(self.feature_names))))
        dice_results = self.generate_counterfactuals(
            data[:5], 
            target_indices,
            desired_outcome='decrease',
            num_counterfactuals=3
        )
        self.visualize_counterfactuals(dice_results, os.path.join(report_dir, 'dice/'))
        
        # Create summary report
        summary = self._create_xai_summary_report(
            shap_results, attention_results, lime_results, 
            causal_results, dice_results, report_dir
        )
        
        print("\n" + "="*80)
        print("XAI REPORT GENERATION COMPLETE")
        print(f"Report saved to: {report_dir}")
        print("="*80)
        
        return summary
    
    def _create_xai_summary_report(self, shap_results: Dict, attention_results: Dict,
                                  lime_results: Dict, causal_results: Dict,
                                  dice_results: Dict, report_dir: str) -> Dict:
        """Create a summary report of all XAI analyses."""
        
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'num_features': len(self.feature_names),
                'config': self.config
            },
            'shap_summary': {
                'global_top_features': list(shap_results.get('global_importance', {}).keys())[:10],
                'num_samples_analyzed': len(shap_results.get('test_data', []))
            } if shap_results else {},
            'attention_summary': {
                'num_layers_analyzed': attention_results.get('num_layers', 0),
                'patch_info': attention_results.get('patch_info', {})
            } if attention_results else {},
            'lime_summary': {
                'num_samples_explained': len(lime_results.get('explanations', [])),
                'top_temporal_patterns': self._extract_lime_patterns(lime_results)
            } if lime_results else {},
            'causal_summary': {
                'num_targets_analyzed': len(causal_results.get('causal_results', {})),
                'top_causal_relationships': self._extract_top_causal(causal_results)
            } if causal_results else {},
            'dice_summary': {
                'num_counterfactuals': len(dice_results.get('counterfactuals', [])),
                'target_features': dice_results.get('target_names', []),
                'desired_outcome': dice_results.get('desired_outcome', '')
            } if dice_results else {},
            'key_insights': self._generate_key_insights(
                shap_results, causal_results, dice_results
            )
        }
        
        # Save summary as JSON
        summary_file = os.path.join(report_dir, 'xai_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create markdown report
        self._create_markdown_report(summary, report_dir)
        
        return summary
    
    def _extract_lime_patterns(self, lime_results: Dict) -> List[str]:
        """Extract top temporal patterns from LIME results."""
        patterns = []
        
        if 'explanations' in lime_results:
            # Count occurrences of time steps in top features
            time_step_counts = defaultdict(int)
            
            for exp in lime_results['explanations']:
                for feature, _ in exp.get('feature_importance', [])[:5]:
                    # Extract time step from feature name
                    if '_t' in feature:
                        try:
                            time_step = int(feature.split('_t')[-1])
                            time_step_counts[time_step] += 1
                        except:
                            pass
            
            # Get top time steps
            top_time_steps = sorted(time_step_counts.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:5]
            
            patterns = [f"Time step {t} (appears {c} times)" 
                       for t, c in top_time_steps]
        
        return patterns
    
    def _extract_top_causal(self, causal_results: Dict) -> List[Dict]:
        """Extract top causal relationships."""
        top_relationships = []
        
        if 'structural_model' in causal_results:
            relationships = causal_results['structural_model'].get('relationships', [])
            
            # Sort by absolute strength
            sorted_rel = sorted(relationships, 
                              key=lambda x: abs(x['strength']), 
                              reverse=True)[:10]
            
            top_relationships = sorted_rel
        
        return top_relationships
    
    def _generate_key_insights(self, shap_results: Dict, 
                              causal_results: Dict, 
                              dice_results: Dict) -> List[str]:
        """Generate key insights from XAI analyses."""
        insights = []
        
        # Insights from SHAP
        if shap_results and 'global_importance' in shap_results:
            top_features = list(shap_results['global_importance'].keys())[:3]
            insights.append(f"Top influential features (SHAP): {', '.join(top_features)}")
        
        # Insights from causal analysis
        if causal_results and 'causal_results' in causal_results:
            causal_targets = list(causal_results['causal_results'].keys())[:3]
            insights.append(f"Key causal targets analyzed: {', '.join(causal_targets)}")
        
        # Insights from DiCE
        if dice_results and 'target_names' in dice_results:
            targets = dice_results['target_names']
            outcome = dice_results.get('desired_outcome', 'unknown')
            insights.append(f"Counterfactuals generated to {outcome} targets: {', '.join(targets)}")
        
        # General insights
        insights.append(f"Model analyzes {len(self.feature_names)} security-related features")
        insights.append("Attention visualization reveals temporal focus patterns")
        insights.append("LIME provides local, interpretable explanations for individual predictions")
        
        return insights
    
    def _create_markdown_report(self, summary: Dict, report_dir: str):
        """Create a markdown report from XAI summary."""
        report_file = os.path.join(report_dir, 'xai_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# XAI Analysis Report\n\n")
            f.write(f"**Generated on:** {summary['timestamp']}\n\n")
            
            f.write("## Model Information\n")
            f.write(f"- Number of features: {summary['model_info']['num_features']}\n")
            f.write(f"- Configuration: {json.dumps(summary['model_info']['config'], indent=2)}\n\n")
            
            if 'shap_summary' in summary and summary['shap_summary']:
                f.write("## SHAP Analysis\n")
                f.write(f"- Top global features: {', '.join(summary['shap_summary']['global_top_features'])}\n")
                f.write(f"- Samples analyzed: {summary['shap_summary']['num_samples_analyzed']}\n\n")
            
            if 'attention_summary' in summary and summary['attention_summary']:
                f.write("## Attention Analysis\n")
                f.write(f"- Transformer layers analyzed: {summary['attention_summary']['num_layers_analyzed']}\n")
                patch_info = summary['attention_summary'].get('patch_info', {})
                if patch_info:
                    f.write(f"- Patch size: {patch_info.get('patch_size', 'N/A')}\n")
                    f.write(f"- Number of patches: {patch_info.get('num_patches', 'N/A')}\n\n")
            
            if 'lime_summary' in summary and summary['lime_summary']:
                f.write("## LIME Analysis\n")
                f.write(f"- Samples explained: {summary['lime_summary']['num_samples_explained']}\n")
                f.write("- Top temporal patterns:\n")
                for pattern in summary['lime_summary'].get('top_temporal_patterns', []):
                    f.write(f"  - {pattern}\n")
                f.write("\n")
            
            if 'causal_summary' in summary and summary['causal_summary']:
                f.write("## Causal Analysis\n")
                f.write(f"- Targets analyzed: {summary['causal_summary']['num_targets_analyzed']}\n")
                f.write("- Top causal relationships:\n")
                for rel in summary['causal_summary'].get('top_causal_relationships', [])[:5]:
                    f.write(f"  - {rel['from']} → {rel['to']} ({rel['type']}, strength: {rel['strength']:.3f})\n")
                f.write("\n")
            
            if 'dice_summary' in summary and summary['dice_summary']:
                f.write("## Counterfactual Analysis (DiCE)\n")
                f.write(f"- Counterfactuals generated: {summary['dice_summary']['num_counterfactuals']}\n")
                f.write(f"- Target features: {', '.join(summary['dice_summary']['target_features'])}\n")
                f.write(f"- Desired outcome: {summary['dice_summary']['desired_outcome']}\n\n")
            
            if 'key_insights' in summary:
                f.write("## Key Insights\n")
                for insight in summary['key_insights']:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            f.write("## Files Generated\n")
            f.write("- SHAP visualizations: `shap/` directory\n")
            f.write("- Attention visualizations: `attention/` directory\n")
            f.write("- LIME explanations: `lime/` directory\n")
            f.write("- Causal analysis: `causal/` directory\n")
            f.write("- Counterfactuals: `dice/` directory\n")
            f.write("- Summary files: `xai_summary.json`, `xai_report.md`\n")
        
        print(f"Markdown report saved to: {report_file}")



# MAIN FUNCTION WITH XAI INTEGRATION


def main_with_xai():
    """Main forecasting function with XAI integration."""
    set_random_seed(123)
    
    # File paths
    model_file = 'final_transfer_model/final_model.pt'
    data_file = './data/sm_data_g.csv'
    nodes_file = './data/sm_data_g.csv'
    graph_file = './data/graph.csv'
    
    print("=" * 80)
    print("FORECASTING WITH XAI INTEGRATION")
    print("=" * 80)
    
    try:
        # Load model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model, config = load_model_and_config(model_file)
        
        # Prepare data
        print("\nPreparing forecast data...")
        data, data_normalized, data_mean, data_std = prepare_forecast_data(data_file, config)
        
        # Get feature names
        col_names, index = create_columns(nodes_file)
        
        # Initialize XAI explainer
        print("\nInitializing XAI explainer...")
        xai_explainer = XAIExplainer(model, config, col_names)
        
    
        # Generate XAI report
        print("\nGenerating comprehensive XAI report...")
        
        # Use a subset of data for XAI analysis to save time
        xai_data = data_normalized[:min(200, len(data_normalized))]
        
        # Ensure proper shape for XAI (batch_size, seq_len, num_nodes)
        if len(xai_data.shape) == 2:
            # Add sequence dimension if needed
            seq_len = config.get('sequence_length', 12)
            xai_data_reshaped = []
            for i in range(len(xai_data) - seq_len):
                xai_data_reshaped.append(xai_data[i:i+seq_len])
            xai_data = np.array(xai_data_reshaped)
        
        # Generate XAI report
        xai_report = xai_explainer.generate_comprehensive_report(
            xai_data, 
            report_dir='model/ViT/xai/full_report/'
        )
        
        
        print(f"\n{'='*80}")
        print("XAI ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"XAI report saved to: model/ViT/xai/full_report/")
        print(f"Individual XAI analyses saved in respective subdirectories")
        
       
        
        return {
            'xai_report': xai_report,
            'config': config,
            'feature_names': col_names
        }
        
    except Exception as e:
        print(f"\nError during XAI analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# STANDALONE XAI ANALYSIS FUNCTION

def run_xai_analysis_only():
    """Run only XAI analysis without forecasting plots."""
    set_random_seed(123)
    
    # File paths
    model_file = 'Dissertation/Vision Transformer/final_transfer_model/final_model.pt'
    data_file = 'Dissertation/Vision Transformer/data/sm_data_g.csv'
    nodes_file = 'Dissertation/Vision Transformer/data/sm_data_g.csv'
    
    print("=" * 80)
    print("STANDALONE XAI ANALYSIS")
    print("=" * 80)
    
    try:
        # Load model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model, config = load_model_and_config(model_file)
        
        # Prepare data
        print("\nPreparing data for XAI analysis...")
        data, data_normalized, data_mean, data_std = prepare_forecast_data(data_file, config)
        
        # Get feature names
        col_names, _ = create_columns(nodes_file)
        
        # Initialize XAI explainer
        print("\nInitializing XAI explainer...")
        xai_explainer = XAIExplainer(model, config, col_names)
        
        
        # Prepare data for XAI
        # Use recent data for analysis
        seq_len = config.get('sequence_length', 12)
        xai_data = []
        
        # Create sequences for XAI
        for i in range(len(data_normalized) - seq_len):
            xai_data.append(data_normalized[i:i+seq_len])
        
        xai_data = np.array(xai_data[-100:])  # Use last 100 sequences
        
        print(f"XAI data shape: {xai_data.shape}")
        
        # Run comprehensive XAI analysis
        xai_report = xai_explainer.generate_comprehensive_report(
            xai_data, 
            report_dir='model/ViT/xai/standalone_report/'
        )
        
        print(f"\n{'='*80}")
        print("STANDALONE XAI ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"Report saved to: model/ViT/xai/standalone_report/")
        
        return xai_report
        
    except Exception as e:
        print(f"\nError during standalone XAI analysis: {e}")
        import traceback
        traceback.print_exc()
        return None



# MODULE-LEVEL XAI FUNCTIONS


def explain_prediction(model: nn.Module, config: dict, input_data: np.ndarray,
                      feature_names: List[str], method: str = 'all') -> Dict:
    """
    High-level function to explain a single prediction.
    
    Args:
        model: Trained model
        config: Model configuration
        input_data: Input data for prediction (seq_len, n_features)
        feature_names: List of feature names
        method: XAI method to use ('shap', 'lime', 'attention', 'causal', 'dice', 'all')
    
    Returns:
        Dictionary with explanations
    """
    # Ensure input has batch dimension
    if len(input_data.shape) == 2:
        input_data = input_data[np.newaxis, ...]
    
    # Initialize explainer
    explainer = XAIExplainer(model, config, feature_names)
    
    results = {}
    
    if method in ['shap', 'all']:
        print("Running SHAP analysis...")
        shap_results = explainer.compute_shap_values(input_data, num_samples=1)
        results['shap'] = shap_results
    
    if method in ['attention', 'all']:
        print("Extracting attention maps...")
        attention_results = explainer.extract_attention_maps(input_data)
        results['attention'] = attention_results
    
    if method in ['lime', 'all']:
        print("Computing LIME explanation...")
        lime_results = explainer.compute_lime_explanations(input_data, num_samples=1)
        results['lime'] = lime_results
    
    if method in ['causal', 'all']:
        print("Performing causal analysis...")
        # For causal analysis, we need more data
        # Use input_data as part of larger dataset
        causal_results = explainer.analyze_causal_relationships(
            input_data[0],  # Remove batch dimension
            target_indices=list(range(min(5, len(feature_names))))
        )
        results['causal'] = causal_results
    
    return results


if __name__ == "__main__":
    # Choose which mode to run
    run_mode = 'xai_only'  # Options: 'full', 'xai_only', 'explain_single'
    
    if run_mode == 'full':
        # Run full forecasting with XAI integration
        results = main_with_xai()
        
    elif run_mode == 'xai_only':
        # Run only XAI analysis
        xai_report = run_xai_analysis_only()
        
    else:
        print(f"Unknown run mode: {run_mode}")
        print("Available modes: 'full', 'xai_only', 'explain_single'")