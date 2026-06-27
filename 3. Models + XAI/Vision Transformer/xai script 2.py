import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
from lime import lime_tabular
import dice_ml
from dice_ml import Dice
import networkx as nx
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class PatchEmbedding(nn.Module):
    """Convert time series into patches with proper size handling"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        if seq_len % patch_size != 0:
            divisors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
            if divisors:
                patch_size = max(divisors)
            else:
                patch_size = 1
        
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x

class VisionTransformerForTimeSeries(nn.Module):
    """Vision Transformer adapted for time series forecasting"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        actual_patch_size = self._get_compatible_patch_size(config['sequence_length'], config['patch_size'])
        actual_num_patches = config['sequence_length'] // actual_patch_size
        
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
        
        self.forecast_head = nn.Sequential(
            nn.Linear(config['embed_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['forecast_horizon'] * config['num_nodes'])
        )
        
        self._init_weights()
        self.mc_dropout = nn.Dropout(config['mc_dropout'])
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def _get_compatible_patch_size(self, seq_len, desired_patch_size):
        divisors = []
        for i in range(1, seq_len + 1):
            if seq_len % i == 0:
                divisors.append(i)
        if not divisors:
            return 1
        return max(divisors)
    
    def forward(self, x, mc_dropout=True):
        x = self.patch_embed(x)
        x = self.transformer(x)
        cls_token = x[:, 0]
        if mc_dropout:
            cls_token = self.mc_dropout(cls_token)
        forecast = self.forecast_head(cls_token)
        forecast = forecast.view(-1, self.config['forecast_horizon'], self.config['num_nodes'])
        return forecast

# XAI IMPLEMENTATIONS

class XAI_Analyzer:
    def __init__(self, model, config, data, feature_names):
        """
        Initialize XAI analyzer for the ViT time series model.
        
        Args:
            model: Trained VisionTransformerForTimeSeries model
            config: Model configuration dictionary
            data: Training data for reference distributions
            feature_names: Names of features/nodes

        """
        self.model = model
        self.config = config
        self.data = data
        self.feature_names = feature_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directories
        self.output_dirs = {
            'shap': 'model/ViT/xai/shap/',
            'lime': 'model/ViT/xai/lime/',
            'dice': 'model/ViT/xai/dice/',
            'attention': 'model/ViT/xai/attention/',
            'causal': 'model/ViT/xai/causal/',
            'feature_importance': 'model/ViT/xai/feature_importance/'
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize categorical features/names
        self.categorical_features = getattr(self, "categorical_features", []) or []
        self.categorical_names = getattr(self, "categorical_names", {}) or {}
    
    def prepare_sample(self, sample_idx=None):
        """Prepare a sample for XAI analysis."""
        if sample_idx is None:
            sample_idx = np.random.randint(0, len(self.data) - self.config['sequence_length'])
        
        # Get sequence
        seq_start = sample_idx
        seq_end = seq_start + self.config['sequence_length']
        sequence = self.data[seq_start:seq_end]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        return sequence_tensor, sequence, seq_start, seq_end
    
    # SHAP ANALYSIS
    
    def shap_analysis(self, n_samples=100):
        """Perform SHAP analysis on the model."""
        print("Performing SHAP analysis...")
        
        # Prepare background data (random samples from training data)
        background_size = min(50, len(self.data) - self.config['sequence_length'])
        background_indices = np.random.choice(
            len(self.data) - self.config['sequence_length'], 
            background_size, 
            replace=False
        )
        
        background_data = []
        for idx in background_indices:
            seq = self.data[idx:idx + self.config['sequence_length']]
            background_data.append(seq)
        
        background_tensor = torch.FloatTensor(np.array(background_data)).to(self.device)
        
        # Define prediction function for SHAP
        def predict_func(X):
            """Wrapper function for model predictions."""
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                X_tensor = X
                
            with torch.no_grad():
                predictions = self.model(X_tensor, mc_dropout=False)
                # Return predictions for first time step of first feature
                return predictions[:, 0, 0].cpu().numpy()
        
        background_np = background_tensor.cpu().numpy()
        
        explainer = None
        shap_values = None

        if isinstance(self.model, torch.nn.Module):
            try:
                self.model.eval()
                background_tensor = torch.from_numpy(background_np).to(self.device)
                explainer = shap.DeepExplainer(self.model, background_tensor)
                shap_values = explainer.shap_values(background_tensor)
                shap_values = [sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv for sv in shap_values]
                # DeepExplainer returns a list per output; handle single-output convenience
                if len(shap_values) == 1:
                    shap_values = shap_values[0]
            except (ValueError, TypeError) as exc:
                print(f"[SHAP] DeepExplainer unsupported ({exc}); falling back to KernelExplainer.")
                explainer = None
                shap_values = None

        if explainer is None:
            explainer = shap.KernelExplainer(predict_func, background_np)
            shap_values = explainer.shap_values(background_np, nsamples=min(100, background_np.shape[0] * 2))
        
        # Get explanations for a few samples
        test_indices = np.random.choice(
            len(self.data) - self.config['sequence_length'], 
            min(5, n_samples), 
            replace=False
        )
        
        for i, idx in enumerate(test_indices):
            test_sequence = self.data[idx:idx + self.config['sequence_length']]
            test_tensor = torch.FloatTensor(test_sequence).unsqueeze(0).to(self.device)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(test_tensor)
            
            # Convert to numpy for visualization
            shap_values_np = shap_values[0].cpu().numpy() if hasattr(shap_values[0], 'cpu') else shap_values[0]
            test_sequence_np = test_sequence.cpu().numpy() if hasattr(test_sequence, 'cpu') else test_sequence
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values_np.reshape(-1, self.config['sequence_length'] * self.config['num_nodes'])[:100],
                test_sequence_np.reshape(-1, self.config['sequence_length'] * self.config['num_nodes'])[:100],
                feature_names=[f"{feat}_t-{t}" for feat in self.feature_names for t in range(self.config['sequence_length'])][:self.config['sequence_length'] * self.config['num_nodes']],
                show=False,
                max_display=20
            )
            plt.title(f"SHAP Summary Plot - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['shap']}shap_summary_sample_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create force plot for the first prediction
            plt.figure(figsize=(15, 3))
            shap.force_plot(
                explainer.expected_value.cpu().numpy() if hasattr(explainer.expected_value, 'cpu') else explainer.expected_value,
                shap_values_np.reshape(-1)[:self.config['sequence_length'] * self.config['num_nodes']],
                test_sequence_np.reshape(-1)[:self.config['sequence_length'] * self.config['num_nodes']],
                feature_names=[f"{feat}_t-{t}" for feat in self.feature_names for t in range(self.config['sequence_length'])][:self.config['sequence_length'] * self.config['num_nodes']],
                show=False,
                matplotlib=True
            )
            plt.title(f"SHAP Force Plot - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['shap']}shap_force_sample_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"SHAP analysis saved to {self.output_dirs['shap']}")
    
    # LIME ANALYSIS 
    
    def lime_analysis(self, n_samples=5):
        """Perform LIME analysis on the model."""
        print("Performing LIME analysis...")
        
        data_flat = self.data[:1000].flatten()  # Adjust the size as needed
        training_data = data_flat.reshape(-1, self.config['num_nodes'])
        
        # Create LIME explainer
        raw_categorical_features = getattr(self, "categorical_features", None) or []
        raw_categorical_names = getattr(self, "categorical_names", {}) or {}
        num_features = training_data.shape[1]
        lime_categorical_features = [
            idx for idx in raw_categorical_features
            if 0 <= idx < num_features
        ]
        if raw_categorical_features and len(lime_categorical_features) != len(raw_categorical_features):
            print(f"[LIME] Dropped {len(raw_categorical_features) - len(lime_categorical_features)} categorical indices outside 0-{num_features - 1}")
        lime_categorical_names = {
            idx: raw_categorical_names[idx]
            for idx in lime_categorical_features
            if idx in raw_categorical_names
        }
        if not lime_categorical_features:
            lime_categorical_features = None
            lime_categorical_names = None
            
        explainer_kwargs = {
            'training_data': training_data,
            'feature_names': self.feature_names,
            'categorical_features': lime_categorical_features,
            'categorical_names': lime_categorical_names,
            'verbose': True,
            'mode': 'regression'
        }
        explainer = lime_tabular.LimeTabularExplainer(**explainer_kwargs)
        # Define prediction function for LIME
        def predict_proba(X):
            """Wrapper function for model predictions."""
            X_tensor = torch.FloatTensor(X).reshape(-1, self.config['sequence_length'], self.config['num_nodes']).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor, mc_dropout=False)
                # Return predictions for first time step
                pred_vals = predictions[:, 0, :].cpu().numpy()
                # Convert to probability-like format for LIME
                proba = np.column_stack([1 - pred_vals[:, 0], pred_vals[:, 0]])
                return proba
        
        # Prepare data for LIME
        data_flat = self.data.reshape(-1, self.config['num_nodes'])
        
        # Limit LIME categorical indices to existing columns
        num_features = data_flat.shape[1]
        lime_categorical_features = [
            idx for idx in (self.categorical_features or [])
            if 0 <= idx < num_features
        ]
        if self.categorical_features and len(lime_categorical_features) != len(self.categorical_features):
            print(f"[LIME] Dropped {len(self.categorical_features) - len(lime_categorical_features)} categorical indices outside 0-{num_features - 1}")
        lime_categorical_names = {
            idx: self.categorical_names[idx]
            for idx in lime_categorical_features
            if idx in self.categorical_names
        }
        if not lime_categorical_features:
            lime_categorical_features = None
            lime_categorical_names = None

       
        
        # Get explanations for a few samples
        test_indices = np.random.choice(len(data_flat) - self.config['sequence_length'], n_samples, replace=False)
        
        for i, idx in enumerate(test_indices):
            # Get test instance
            test_instance = data_flat[idx:idx + self.config['sequence_length']].flatten()
            
            # Get LIME explanation
            try:
                exp = explainer.explain_instance(
                    test_instance,
                    predict_proba,
                    num_features=min(10, len(self.feature_names) * self.config['sequence_length']),
                    num_samples=500
                )
            except KeyError as exc:
                if lime_categorical_features:
                    print(f"[LIME] Categorical lookup failed ({exc}); retrying without categorical metadata.")
                    explainer_kwargs['categorical_features'] = None
                    explainer_kwargs['categorical_names'] = None
                    explainer = lime.lime_tabular.LimeTabularExplainer(**explainer_kwargs)
                    exp = explainer.explain_instance(
                        test_instance,
                        predict_proba,
                        num_features=min(10, len(self.feature_names) * self.config['sequence_length']),
                        num_samples=500
                    )
                else:
                    raise
            
            # Save explanation as HTML
            exp.save_to_file(f"{self.output_dirs['lime']}lime_explanation_sample_{i+1}.html")
            
            # Also save as image
            fig = exp.as_pyplot_figure()
            plt.title(f"LIME Explanation - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['lime']}lime_plot_sample_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance from LIME
            lime_weights = dict(exp.as_list())
            with open(f"{self.output_dirs['lime']}lime_weights_sample_{i+1}.json", 'w') as f:
                json.dump(lime_weights, f, indent=2)
        
        print(f"LIME analysis saved to {self.output_dirs['lime']}")
    
    #  DiCE COUNTERFACTUALS 
    
    def dice_counterfactuals(self, n_counterfactuals=3):
        """Generate counterfactual explanations using DiCE."""
        print("Generating DiCE counterfactuals...")
        
        # Prepare data for DiCE
        df_data = pd.DataFrame(self.data[:1000], columns=self.feature_names)  # Use subset
        
        # Define prediction function for DiCE
        def dice_predict_func(df):
            """Prediction function for DiCE."""
            # Convert dataframe to tensor
            data_tensor = torch.FloatTensor(df.values).reshape(
                -1, 1, len(self.feature_names)
            ).repeat(1, self.config['sequence_length'], 1).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(data_tensor, mc_dropout=False)
                return predictions[:, 0, :].cpu().numpy()
        
        # Create DiCE data object
        data_dice = dice_ml.Data(
            dataframe=df_data,
            continuous_features=self.feature_names,
            outcome_name='prediction'
        )
        
        # Create DiCE model object
        model_dice = dice_ml.Model(
            model=dice_predict_func,
            backend='PYT',
            model_type='regressor'
        )
        
        # Create DiCE explainer
        explainer = Dice(data_dice, model_dice, method='random')
        
        # Generate counterfactuals for a few instances
        test_indices = np.random.choice(len(df_data), min(3, len(df_data)), replace=False)
        
        for i, idx in enumerate(test_indices):
            query_instance = df_data.iloc[[idx]]
            
            # Generate counterfactuals
            dice_exp = explainer.generate_counterfactuals(
                query_instance,
                total_CFs=n_counterfactuals,
                desired_range=[0.1, 0.3]  # Desired prediction range
            )
            
            # Visualize counterfactuals
            plt.figure(figsize=(12, 6))
            dice_exp.visualize_as_dataframe(show=False)
            plt.title(f"DiCE Counterfactuals - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['dice']}dice_counterfactuals_sample_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save counterfactuals to CSV
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            cf_df.to_csv(f"{self.output_dirs['dice']}dice_counterfactuals_sample_{i+1}.csv", index=False)
            
            # Plot comparison
            self._plot_counterfactual_comparison(query_instance, cf_df, i+1)
        
        print(f"DiCE counterfactuals saved to {self.output_dirs['dice']}")
    
    def _plot_counterfactual_comparison(self, original, counterfactuals, sample_id):
        """Plot comparison between original and counterfactual instances."""
        plt.figure(figsize=(15, 5))
        
        # Plot original features
        plt.subplot(1, 2, 1)
        original_values = original.values.flatten()
        plt.bar(range(len(self.feature_names)), original_values[:len(self.feature_names)])
        plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=90)
        plt.title(f"Original Instance - Sample {sample_id}")
        plt.ylabel("Feature Value")
        plt.tight_layout()
        
        # Plot counterfactual changes
        plt.subplot(1, 2, 2)
        for cf_idx in range(min(3, len(counterfactuals))):
            cf_values = counterfactuals.iloc[cf_idx].values.flatten()
            changes = cf_values[:len(self.feature_names)] - original_values[:len(self.feature_names)]
            plt.bar(range(len(self.feature_names)), changes, alpha=0.7, label=f'CF {cf_idx+1}')
        
        plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=90)
        plt.title(f"Counterfactual Changes - Sample {sample_id}")
        plt.ylabel("Change from Original")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{self.output_dirs['dice']}dice_comparison_sample_{sample_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ATTENTION VISUALIZATION 
    
    def attention_visualization(self, n_samples=3):
        """Visualize attention patterns in the transformer."""
        print("Visualizing attention patterns...")
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # Extract attention weights from transformer layer
            # Assuming self-attention is the first module in the layer
            for layer in module.layers:
                attn_module = layer.self_attn
                # Get attention weights (if available)
                if hasattr(attn_module, 'attn_weight'):
                    attention_weights.append(attn_module.attn_weight.detach().cpu())
        
        # Register hook
        hook = self.model.transformer.register_forward_hook(attention_hook)
        
        # Process a few samples
        test_indices = np.random.choice(len(self.data) - self.config['sequence_length'], n_samples, replace=False)
        
        for i, idx in enumerate(test_indices):
            attention_weights.clear()
            
            # Get test sequence
            test_sequence = self.data[idx:idx + self.config['sequence_length']]
            test_tensor = torch.FloatTensor(test_sequence).unsqueeze(0).to(self.device)
            
            # Forward pass to capture attention
            with torch.no_grad():
                _ = self.model(test_tensor, mc_dropout=False)
            
            # Visualize attention weights if captured
            if attention_weights:
                for layer_idx, attn in enumerate(attention_weights):
                    if attn is not None:
                        # Average over heads
                        attn_mean = attn.mean(dim=1).squeeze().numpy()
                        
                        # Plot attention matrix
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            attn_mean,
                            cmap='viridis',
                            xticklabels=[f"Patch {j}" for j in range(attn_mean.shape[1])],
                            yticklabels=[f"Patch {j}" for j in range(attn_mean.shape[0])]
                        )
                        plt.title(f"Attention Heatmap - Sample {i+1}, Layer {layer_idx+1}")
                        plt.xlabel("Key Patches")
                        plt.ylabel("Query Patches")
                        plt.tight_layout()
                        plt.savefig(f"{self.output_dirs['attention']}attention_sample_{i+1}_layer_{layer_idx+1}.png", 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Plot attention to CLS token
                        plt.figure(figsize=(12, 6))
                        cls_attention = attn_mean[0, 1:]  # CLS token attends to other patches
                        plt.bar(range(len(cls_attention)), cls_attention)
                        plt.title(f"CLS Token Attention - Sample {i+1}, Layer {layer_idx+1}")
                        plt.xlabel("Patch Index")
                        plt.ylabel("Attention Weight")
                        plt.tight_layout()
                        plt.savefig(f"{self.output_dirs['attention']}cls_attention_sample_{i+1}_layer_{layer_idx+1}.png", 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
        
        # Remove hook
        hook.remove()
        
        print(f"Attention visualizations saved to {self.output_dirs['attention']}")
    
    # CAUSAL NETWORK ANALYSIS
    
    def causal_analysis(self):
        """Perform causal network analysis."""
        print("Performing causal network analysis...")
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(self.data.T)
        
        # Create causal graph based on correlations
        G = nx.Graph()
        
        # Add nodes
        for feature in self.feature_names:
            G.add_node(feature)
        
        # Add edges based on correlation threshold
        threshold = 0.3
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                if abs(correlation_matrix[i, j]) > threshold:
                    G.add_edge(
                        self.feature_names[i], 
                        self.feature_names[j],
                        weight=correlation_matrix[i, j]
                    )
        
        # Plot causal network
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
        
        # Draw edges with width proportional to correlation
        edges = G.edges(data=True)
        widths = [abs(d['weight']) * 5 for (u, v, d) in edges]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title("Causal Network Analysis (Based on Correlations)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_dirs['causal']}causal_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate network metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Save network metrics
        metrics = {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'number_of_nodes': G.number_of_nodes(),
            'number_of_edges': G.number_of_edges()
        }
        
        with open(f"{self.output_dirs['causal']}network_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot centrality measures
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Degree centrality
        degrees = list(degree_centrality.values())
        axes[0].bar(range(len(degrees)), sorted(degrees, reverse=True))
        axes[0].set_title("Degree Centrality Distribution")
        axes[0].set_xlabel("Nodes (sorted)")
        axes[0].set_ylabel("Degree Centrality")
        
        # Betweenness centrality
        betweenness = list(betweenness_centrality.values())
        axes[1].bar(range(len(betweenness)), sorted(betweenness, reverse=True))
        axes[1].set_title("Betweenness Centrality Distribution")
        axes[1].set_xlabel("Nodes (sorted)")
        axes[1].set_ylabel("Betweenness Centrality")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dirs['causal']}centrality_measures.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Causal analysis saved to {self.output_dirs['causal']}")
    
    # FEATURE IMPORTANCE
    
    def feature_importance_analysis(self, n_iterations=100):
        """Perform feature importance analysis using permutation importance."""
        print("Performing feature importance analysis...")
        
        # Prepare baseline predictions
        test_indices = np.random.choice(
            len(self.data) - self.config['sequence_length'], 
            min(50, n_iterations), 
            replace=False
        )
        
        baseline_predictions = []
        for idx in tqdm(test_indices, desc="Baseline predictions"):
            test_sequence = self.data[idx:idx + self.config['sequence_length']]
            test_tensor = torch.FloatTensor(test_sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(test_tensor, mc_dropout=False)
                baseline_predictions.append(pred[:, 0, :].cpu().numpy())
        
        baseline_mean = np.mean(np.concatenate(baseline_predictions, axis=0), axis=0)
        
        # Calculate permutation importance for each feature
        importance_scores = np.zeros((len(self.feature_names), len(self.feature_names)))
        
        for feature_idx in tqdm(range(len(self.feature_names)), desc="Permutation importance"):
            permuted_predictions = []
            
            for idx in test_indices:
                test_sequence = self.data[idx:idx + self.config['sequence_length']].copy()
                
                # Permute the feature across time dimension
                permuted_sequence = test_sequence.copy()
                permuted_sequence[:, feature_idx] = np.random.permutation(permuted_sequence[:, feature_idx])
                
                test_tensor = torch.FloatTensor(permuted_sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred = self.model(test_tensor, mc_dropout=False)
                    permuted_predictions.append(pred[:, 0, :].cpu().numpy())
            
            permuted_mean = np.mean(np.concatenate(permuted_predictions, axis=0), axis=0)
            
            # Calculate importance as change in prediction
            importance_scores[feature_idx] = np.abs(baseline_mean - permuted_mean)
        
        # Plot feature importance heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            importance_scores,
            cmap='YlOrRd',
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Importance Score'}
        )
        plt.title("Feature Importance Matrix (Permutation Importance)")
        plt.xlabel("Predicted Feature")
        plt.ylabel("Permuted Feature")
        plt.tight_layout()
        plt.savefig(f"{self.output_dirs['feature_importance']}feature_importance_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot top important features for each prediction
        top_n = min(10, len(self.feature_names))
        
        for pred_idx in range(len(self.feature_names)):
            feature_importance = importance_scores[:, pred_idx]
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(top_n), feature_importance[top_indices])
            plt.xticks(range(top_n), [self.feature_names[i] for i in top_indices], rotation=45, ha='right')
            plt.title(f"Top {top_n} Important Features for Predicting {self.feature_names[pred_idx]}")
            plt.ylabel("Importance Score")
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['feature_importance']}importance_for_{self.feature_names[pred_idx].replace('/', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save importance scores
        importance_df = pd.DataFrame(
            importance_scores,
            index=[f"Permuted_{feat}" for feat in self.feature_names],
            columns=[f"Predict_{feat}" for feat in self.feature_names]
        )
        importance_df.to_csv(f"{self.output_dirs['feature_importance']}feature_importance_scores.csv")
        
        print(f"Feature importance analysis saved to {self.output_dirs['feature_importance']}")
    
    # RUN ALL ANALYSES
    
    def run_all_analyses(self):
        """Run all XAI analyses."""
        print("=" * 80)
        print("STARTING COMPREHENSIVE XAI ANALYSIS")
        print("=" * 80)
        
        # Run each analysis
        
        self.shap_analysis(n_samples=50)
        print("-" * 40)
        
        self.lime_analysis(n_samples=5)
        print("-" * 40)
        
        self.dice_counterfactuals(n_counterfactuals=3)
        print("-" * 40)
        
        
        self.attention_visualization(n_samples=3)
        print("-" * 40)
        
        self.causal_analysis()
        print("-" * 40)
        
        self.feature_importance_analysis(n_iterations=50)
        print("-" * 40)
        
        print("=" * 80)
        print("XAI ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nResults saved to:")
        for method, path in self.output_dirs.items():
            print(f"  - {method}: {path}")

# MAIN EXECUTION

def load_model_and_data():
    """Load the trained model and data."""
    # Load model configuration 
    model_path = 'Dissertation/Vision Transformer/final_transfer_model/final_model.pt'
    data_path = 'Dissertation/Vision Transformer/data/sm_data_g.csv'
    
    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_path}")
    
    # Load model 
    try:
        model_data = torch.load(model_path, map_location='cpu')
        config = model_data['hyperparameters']
        state_dict = model_data['model_state_dict']
        
        # Create model
        model = VisionTransformerForTimeSeries(config)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Configuration: {config}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating dummy model for demonstration...")
        # Create dummy config for demonstration
        config = {
            'sequence_length': 24,
            'patch_size': 6,
            'num_nodes': 10,
            'embed_dim': 64,
            'num_heads': 4,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
            'mc_dropout': 0.1,
            'forecast_horizon': 12
        }
        model = VisionTransformerForTimeSeries(config)
    
    # Load data
    data = pd.read_csv(data_path)
    if 'Date' in data.columns or 'date' in data.columns.str.lower():
        data = data.iloc[:, 1:]  # Remove date column
    
    data_values = data.values.astype(np.float32)
    feature_names = data.columns.tolist()
    
    print(f"Data shape: {data_values.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    return model, config, data_values, feature_names

def main():
    """Main function to run XAI analysis."""
    # Load model and data
    model, config, data, feature_names = load_model_and_data()
    
    # Create XAI analyzer
    analyzer = XAI_Analyzer(model, config, data, feature_names)
    
    # Run all analyses
    analyzer.run_all_analyses()
    
    print("\n" + "=" * 80)
    print("XAI ANALYSIS SUMMARY")
    print("=" * 80)
    print("Generated the following analyses:")
    print("- SHAP: Global and local feature importance using SHAP values")
    print("- LIME: Local interpretable model-agnostic explanations")
    print("- DiCE: Counterfactual explanations for 'what-if' scenarios")
    print("- Attention: Visualization of transformer attention patterns")
    print("- Causal: Network analysis of feature relationships")
    print("- Feature Importance: Permutation-based feature importance")
    print("\nAll results saved to: model/ViT/xai/")

if __name__ == "__main__":
    main()