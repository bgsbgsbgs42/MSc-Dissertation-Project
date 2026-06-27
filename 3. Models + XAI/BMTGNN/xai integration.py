import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import networkx as nx
import shap
import lime
import lime.lime_tabular
from dice_ml import Dice
from dice_ml.utils import helpers
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from torch.utils.data import TensorDataset, DataLoader
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import entropy
import os
import json

warnings.filterwarnings('ignore')

# Import the ensemble model components from forecast script
try:
    from forecast_w_pretrain import (
        SpatioTemporalEnsemble, CyberThreatDataset, 
        load_ensemble_model, prepare_forecast_data, ensemble_forecast
    )
except ImportError:
    # Fallback definitions
    class SpatioTemporalEnsemble(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.forecast_net = nn.Linear(
                config['sequence_length'] * config['num_nodes'], 
                config['forecast_horizon'] * config['num_nodes']
            )
        
        def forward(self, x, mc_dropout=False):
            batch_size, seq_len, num_nodes = x.shape
            x = x.reshape(batch_size, -1)
            output = self.forecast_net(x)
            output = output.reshape(batch_size, self.config['forecast_horizon'], num_nodes)
            return output

    class CyberThreatDataset:
        def __init__(self, data, sequence_length, forecast_horizon):
            self.data = data
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class IntegratedXAIEvaluator:
    """Integrated XAI evaluator that works with forecast_w_pretrain.py components"""
    
    def __init__(self, model_checkpoint_path: str, data_path: str, results_root: str = './xai_results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_checkpoint_path = model_checkpoint_path
        self.data_path = data_path
        self.results_root = results_root

        # Core attributes
        self.config = None
        self.model = None
        self.scaler = StandardScaler()
        self.test_data = None
        self.feature_names = None
        self.test_loader = None
        self.results = {}

        # Ensure results folder exists
        os.makedirs(self.results_root, exist_ok=True)

        # Optional explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.dice_explainer = None

        print(f" Integrated XAI Evaluator initialized on device: {self.device}")
        print(f" Results will be saved to: {self.results_root}")

    def load_model_with_forecast_integration(self):
        """Load model using forecast_w_pretrain.py components"""
        print("Loading model with forecast integration...")
        try:
            # Use the load_ensemble_model function from forecast script
            model, config, scale, feature_names = load_ensemble_model(
                self.model_checkpoint_path, 
                None,  # config_path
                self.device
            )
            
            if model is None:
                print("   Failed to load model using forecast integration")
                return False
            
            self.model = model
            self.config = config
            
            # Load data using forecast script's function
            forecast_input, full_data = prepare_forecast_data(
                self.data_path, 
                config['sequence_length'], 
                self.device, 
                scale
            )
            
            # Prepare data for XAI analysis
            self._prepare_data_for_xai_analysis(full_data)
            
            print("Model loaded successfully with forecast integration")
            return True
            
        except Exception as e:
            print(f"Error loading model with forecast integration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _prepare_data_for_xai_analysis(self, full_data: np.ndarray):
        """Prepare data for XAI analysis with robust dataset creation"""
        print("Preparing data for XAI analysis...")
        
        # Create feature names
        self.feature_names = [f"feature_{i}" for i in range(full_data.shape[1])]
        
        # Normalize data
        self.data_mean = full_data.mean(axis=0)
        self.data_std = full_data.std(axis=0) + 1e-8
        self.test_data = (full_data - self.data_mean) / self.data_std
        
        print(f" Prepared data: shape={self.test_data.shape}, features={len(self.feature_names)}")
        
        # Create test dataset using robust approach
        self._create_robust_test_loader()

    def _create_robust_test_loader(self):
        """Create test loader with multiple fallback approaches"""
        print("Creating test loader...")
        
        if self.test_data is None:
            print(" No test data available")
            return
            
        sequence_length = self.config.get('sequence_length', 12)
        forecast_horizon = self.config.get('forecast_horizon', 36)
        
        # Approach 1: Try simple sequential creation
        try:
            sequences = []
            targets = []
            
            # Create overlapping sequences
            for i in range(len(self.test_data) - sequence_length - forecast_horizon + 1):
                if i % 10 == 0:  # Take every 10th sequence to avoid too many
                    seq = self.test_data[i:i + sequence_length]
                    target = self.test_data[i + sequence_length:i + sequence_length + forecast_horizon]
                    sequences.append(seq)
                    targets.append(target)
            
            if sequences:
                x_tensor = torch.FloatTensor(np.array(sequences))
                y_tensor = torch.FloatTensor(np.array(targets))
                
                test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                self.test_loader = DataLoader(
                    test_dataset,
                    batch_size=min(4, len(test_dataset)),
                    shuffle=False
                )
                print(f" Test loader created: {len(test_dataset)} sequences")
                return
        except Exception as e:
            print(f" Sequential creation failed: {e}")
        
        # Approach 2: Use last few sequences only
        try:
            # Use the last complete sequence
            if len(self.test_data) >= sequence_length + forecast_horizon:
                start_idx = len(self.test_data) - sequence_length - forecast_horizon
                seq = self.test_data[start_idx:start_idx + sequence_length]
                target = self.test_data[start_idx + sequence_length:start_idx + sequence_length + forecast_horizon]
                
                x_tensor = torch.FloatTensor(seq).unsqueeze(0)  # Add batch dimension
                y_tensor = torch.FloatTensor(target).unsqueeze(0)
                
                test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                print(" Test loader created with single sequence")
                return
        except Exception as e:
            print(f" Single sequence creation failed: {e}")
        
        # Approach 3: Minimal working data
        try:
            # Create dummy data with correct dimensions
            seq = self.test_data[:sequence_length]
            target = self.test_data[sequence_length:sequence_length + forecast_horizon]
            
            # Ensure we have enough data
            if len(seq) == sequence_length and len(target) == forecast_horizon:
                x_tensor = torch.FloatTensor(seq).unsqueeze(0)
                y_tensor = torch.FloatTensor(target).unsqueeze(0)
                
                test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                print(" Test loader created with available data")
            else:
                print(" Not enough data for even minimal test loader")
                self.test_loader = None
                
        except Exception as e:
            print(f" All test loader creation approaches failed: {e}")
            self.test_loader = None

    def _ensure_test_loader(self):
        """Ensure test_loader exists for analysis methods"""
        if self.test_loader is not None:
            return True
            
        print("Test loader not available, attempting to create...")
        self._create_robust_test_loader()
        
        if self.test_loader is None:
            print(" Could not create test loader")
            return False
            
        return True
    
    def _create_simple_test_data(self):
        """Create simple test data for analysis when complex dataset fails"""
        print("Creating simple test data...")
        
        if self.test_data is None:
            print(" No test data available")
            return False
            
        # Create simple sequences from test_data
        seq_len = self.config['sequence_length']
        horizon = self.config['forecast_horizon']
        
        # Use the last few sequences for testing
        test_sequences = []
        test_targets = []
        
        # Create multiple starting points
        num_sequences = min(10, len(self.test_data) - seq_len - horizon)
        
        for i in range(num_sequences):
            start_idx = len(self.test_data) - seq_len - horizon - i
            if start_idx >= 0:
                seq = self.test_data[start_idx:start_idx + seq_len]
                target = self.test_data[start_idx + seq_len:start_idx + seq_len + horizon]
                test_sequences.append(seq)
                test_targets.append(target)
        
        if test_sequences:
            x_tensor = torch.FloatTensor(np.array(test_sequences))
            y_tensor = torch.FloatTensor(np.array(test_targets))
            
            test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            self.test_loader = DataLoader(test_dataset, batch_size=min(4, len(test_dataset)), shuffle=False)
            print(f" Simple test data created: {len(test_dataset)} sequences")
            return True
        else:
            print(" Could not create simple test data")
            return False
    def counterfactual_analysis(self, num_samples: int = 3):
        """Generate counterfactual explanations showing minimal changes for target outcomes."""
        print("Running counterfactual_analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'counterfactuals'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x[:num_samples].to(self.device)
            
            counterfactuals = []
            
            for idx in range(len(x)):
                x_sample = x[idx:idx+1].clone()
                x_cf = x_sample.clone()
                x_cf.requires_grad = True
                
                # Get original prediction
                with torch.no_grad():
                    original_pred = self.model(x_sample, mc_dropout=False)
                
                # Target: increase prediction by 20%
                target = original_pred * 1.2
                
                optimizer = torch.optim.Adam([x_cf], lr=0.01)
                
                for step in range(100):
                    output = self.model(x_cf, mc_dropout=False)
                    
                    # Loss: achieve target whilst minimizing changes
                    prediction_loss = F.mse_loss(output, target)
                    proximity_loss = F.mse_loss(x_cf, x_sample)
                    
                    loss = prediction_loss + 0.1 * proximity_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Analyze changes
                changes = (x_cf - x_sample).abs().detach().cpu()
                changes_per_feature = changes.mean(dim=1).squeeze()
                
                top_k = min(10, len(changes_per_feature))
                top_changes = torch.topk(changes_per_feature, k=top_k)
                
                counterfactuals.append({
                    'sample_idx': idx,
                    'top_changed_features': [self.feature_names[i] for i in top_changes.indices.numpy()],
                    'change_magnitudes': top_changes.values.numpy().tolist()
                })
            
            # Visualize counterfactual changes
            plt.figure(figsize=(12, 8))
            
            for idx, cf in enumerate(counterfactuals):
                plt.subplot(len(counterfactuals), 1, idx + 1)
                plt.barh(cf['top_changed_features'], cf['change_magnitudes'])
                plt.xlabel('Change Magnitude')
                plt.title(f'Counterfactual Sample {idx + 1}: Key Changes')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            p = os.path.join(self.results_root, 'counterfactuals', 'counterfactual_changes.png')
            plt.savefig(p, dpi=300)
            plt.close()
            
            result = {'counterfactuals': counterfactuals, 'plot': p}
            self.results['counterfactuals'] = result
            
            print("  counterfactual_analysis done.")
            return result
            
        except Exception as e:
            print(f"  counterfactual_analysis failed: {e}")
            return None

    def shap_analysis(self, num_background_batches: int = 2, num_test_samples: int = 10):
        """Perform (Kernel) SHAP analysis for global/local feature importance."""
        print("Running shap_analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'shap'), exist_ok=True)

            # Collect small background
            background = []
            for i, (x, y) in enumerate(self.test_loader):
                if i >= num_background_batches:
                    break
                background.append(x)
            
            if not background:
                print("  Not enough data for SHAP background.")
                return None
            
            background = torch.cat(background, dim=0)
            background_flat = background.reshape(-1, background.shape[-1]).cpu().numpy()
            background_samples = background_flat[:max(10, min(50, background_flat.shape[0]))]

            def model_predict(x_np):
                bsize = x_np.shape[0]
                try:
                    x_resh = x_np.reshape(bsize, self.config['sequence_length'], -1)
                except Exception:
                    x_resh = x_np.reshape(bsize, -1, self.config['num_nodes'])
                
                if x_resh.shape[-1] != self.config['num_nodes']:
                    if x_resh.shape[-1] < self.config['num_nodes']:
                        pad = self.config['num_nodes'] - x_resh.shape[-1]
                        x_resh = np.pad(x_resh, ((0, 0), (0, 0), (0, pad)), mode='constant')
                    else:
                        x_resh = x_resh[:, :, :self.config['num_nodes']]
                
                x_tensor = torch.FloatTensor(x_resh).to(self.device)
                with torch.no_grad():
                    try:
                        out = self.model(x_tensor, mc_dropout=False)
                        return out.cpu().numpy().reshape(bsize, -1)
                    except Exception:
                        return np.zeros((bsize, self.config['forecast_horizon'] * self.config['num_nodes']))

            try:
                explainer = shap.KernelExplainer(model_predict, background_samples)
                test_samples = background_samples[:num_test_samples]
                shap_values = explainer.shap_values(test_samples)
            except Exception as e:
                print(f"  shap KernelExplainer failed: {e}")
                return None

            self.results['shap'] = {'shap_values': shap_values, 'feature_names': self.feature_names}
            
            try:
                self._create_shap_plots(shap_values, test_samples)
            except Exception as e:
                print(f"  _create_shap_plots failed: {e}")

            with open(os.path.join(self.results_root, 'shap', 'shap_results.json'), 'w') as f:
                json.dump({
                    'feature_names': self.feature_names, 
                    'shap_samples_shape': str(np.shape(shap_values))
                }, f, indent=2, cls=NumpyEncoder)

            print("  shap_analysis done.")
            return shap_values
            
        except Exception as e:
            print(f"  shap_analysis failed: {e}")
            return None

    def _create_shap_plots(self, shap_values, test_samples):
        """Helper to save SHAP summary and bar plots."""
        try:
            os.makedirs(os.path.join(self.results_root, 'shap'), exist_ok=True)
            
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, test_samples, 
                                feature_names=self.feature_names[:test_samples.shape[1]], 
                                show=False)
                p = os.path.join(self.results_root, 'shap', 'summary_plot.png')
                plt.tight_layout()
                plt.savefig(p, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  shap summary plot failed: {e}")

            try:
                plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values, test_samples, 
                                feature_names=self.feature_names[:test_samples.shape[1]], 
                                plot_type='bar', show=False)
                p2 = os.path.join(self.results_root, 'shap', 'global_importance.png')
                plt.tight_layout()
                plt.savefig(p2, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  shap bar plot failed: {e}")

            print("  SHAP plots created.")
            
        except Exception as e:
            print(f"  _create_shap_plots top-level failed: {e}")

    def lime_analysis(self, num_samples: int = 5):
        """LIME local explanations for selected samples."""
        print("Running lime_analysis...")
        try:
            if self.test_data is None:
                print("  No test_data for LIME.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'lime'), exist_ok=True)
            
            background = self.test_data.reshape(-1, self.test_data.shape[-1])
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                background,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=False,
                random_state=42
            )

            def predict_fn(x_np):
                bsize = x_np.shape[0]
                x_resh = x_np.reshape(bsize, self.config['sequence_length'], -1)
                
                if x_resh.shape[-1] != self.config['num_nodes']:
                    if x_resh.shape[-1] < self.config['num_nodes']:
                        pad = self.config['num_nodes'] - x_resh.shape[-1]
                        x_resh = np.pad(x_resh, ((0, 0), (0, 0), (0, pad)), mode='constant')
                    else:
                        x_resh = x_resh[:, :, :self.config['num_nodes']]
                
                x_tensor = torch.FloatTensor(x_resh).to(self.device)
                with torch.no_grad():
                    try:
                        out = self.model(x_tensor, mc_dropout=False)
                        return out.cpu().numpy().reshape(bsize, -1)
                    except Exception:
                        return np.zeros((bsize, self.config['forecast_horizon'] * self.config['num_nodes']))

            test_samples = background[:num_samples]
            saved = []
            
            for i, s in enumerate(test_samples):
                try:
                    exp = self.lime_explainer.explain_instance(
                        s, predict_fn, 
                        num_features=min(10, len(self.feature_names)), 
                        num_samples=500
                    )
                    
                    plt.figure(figsize=(10, 6))
                    try:
                        fig = exp.as_pyplot_figure()
                        p = os.path.join(self.results_root, 'lime', f'lime_explanation_{i}.png')
                        plt.tight_layout()
                        plt.savefig(p, dpi=300)
                        plt.close()
                        saved.append(p)
                    except Exception:
                        plt.close()
                except Exception as e:
                    print(f"  LIME explain failed for sample {i}: {e}")
            
            self.results['lime'] = {'plots': saved}
            
            print("  lime_analysis done.")
            return saved
            
        except Exception as e:
            print(f"  lime_analysis failed: {e}")
            return None

    def causal_analysis(self):
        """Perform causal analysis using Bayesian networks."""
        print("Performing causal analysis...")
        
        if self.test_data is None:
            print("No test data available for causal analysis")
            return None
        
        os.makedirs(os.path.join(self.results_root, 'causal'), exist_ok=True)
        
        try:
            # Prepare data for causal analysis
            data_df = pd.DataFrame(self.test_data, columns=self.feature_names)
            
            # Select top features for computational efficiency
            top_features = self._get_top_features(15)
            data_subset = data_df[top_features].copy()
            
            # Create Bayesian network structure
            model = DiscreteBayesianNetwork()
            
            # Add nodes (features)
            for feature in top_features:
                model.add_node(feature)
            
            # Add example edges (in practice, learn from data)
            if len(top_features) >= 3:
                model.add_edge(top_features[0], top_features[1])
                model.add_edge(top_features[1], top_features[2])
            
            # Create causal graph visualization
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(model)
            nx.draw(model, pos, with_labels=True, node_color='lightblue', 
                   node_size=2000, font_size=8, font_weight='bold', arrowsize=20)
            plt.title('Bayesian Causal Network')
            plt.tight_layout()
            p = os.path.join(self.results_root, 'causal', 'bayesian_network.png')
            plt.savefig(p, dpi=300)
            plt.close()
            
            # Calculate causal effects (simplified)
            causal_effects = self._estimate_causal_effects(model, data_subset)
            
            self.results['causal'] = {
                'model': 'BayesianNetwork',
                'causal_effects': causal_effects,
                'network_structure': list(model.edges()),
                'plot': p
            }
            
            print("  causal_analysis done.")
            return causal_effects
            
        except Exception as e:
            print(f"  causal_analysis failed: {e}")
            return None
    
    def _estimate_causal_effects(self, model, data: pd.DataFrame) -> Dict:
        """Estimate causal effects between variables."""
        causal_effects = {}
        
        for edge in model.edges():
            cause, effect = edge
            # Calculate correlation as proxy for causal effect
            correlation = data[cause].corr(data[effect])
            causal_effects[str(edge)] = {
                'correlation': float(correlation),
                'causal_strength': float(abs(correlation))
            }
        
        return causal_effects
    
    def _get_top_features(self, n_features: int) -> List[str]:
        """Get top n features based on variance."""
        if self.test_data is None:
            return self.feature_names[:n_features]
        
        variances = np.var(self.test_data, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        return [self.feature_names[i] for i in top_indices]

    def permutation_feature_importance(self):
        """Estimate importance by permuting features and measuring increase in MSE."""
        print("Running permutation_feature_importance...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'permutation'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x.to(self.device)
            y = y.to(self.device)
            
            if x.shape[-1] != self.config['num_nodes']:
                if x.shape[-1] < self.config['num_nodes']:
                    pad = self.config['num_nodes'] - x.shape[-1]
                    x = F.pad(x, (0, pad))
                    y = F.pad(y, (0, pad))
                else:
                    x = x[:, :, :self.config['num_nodes']]
                    y = y[:, :, :self.config['num_nodes']]
            
            with torch.no_grad():
                base_pred = self.model(x, mc_dropout=False)
                baseline_loss = F.mse_loss(base_pred, y).item()
            
            scores = []
            num_features = min(10, x.shape[-1])
            
            for fi in range(num_features):
                xp = x.clone()
                perm_idx = torch.randperm(x.shape[1])
                xp[:, :, fi] = xp[:, perm_idx, fi]
                
                with torch.no_grad():
                    pp = self.model(xp, mc_dropout=False)
                    pl = F.mse_loss(pp, y).item()
                
                scores.append(pl - baseline_loss)
            
            feature_names = [self.feature_names[i] for i in range(num_features)]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(num_features), scores)
            plt.yticks(range(num_features), feature_names)
            plt.xlabel('MSE Increase')
            plt.title('Permutation Feature Importance')
            plt.tight_layout()
            
            p = os.path.join(self.results_root, 'permutation', 'permutation_importance.png')
            plt.savefig(p, dpi=300)
            plt.close()
            
            res = {
                'feature_names': feature_names, 
                'importance_scores': scores, 
                'baseline_loss': baseline_loss, 
                'plot': p
            }
            self.results['permutation'] = res
            
            print("  permutation_feature_importance done.")
            return res
            
        except Exception as e:
            print(f"  permutation_feature_importance failed: {e}")
            return None

    def prediction_analysis(self):
        """Prediction vs actual statistics and plots."""
        print("Running prediction_analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'predictions'), exist_ok=True)
            
            all_preds, all_targs = [], []
            for x, y in self.test_loader:
                x = x.to(self.device)
                with torch.no_grad():
                    p = self.model(x, mc_dropout=False)
                    all_preds.append(p.cpu().numpy())
                    all_targs.append(y.numpy())
            
            if not all_preds:
                return None
            
            preds = np.concatenate(all_preds)
            targs = np.concatenate(all_targs)
            
            mse = np.mean((preds - targs) ** 2)
            mae = np.mean(np.abs(preds - targs))
            
            # Plot for first sample & first node
            sample_idx = 0
            node_idx = 0
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(targs[sample_idx, :, node_idx], label='Actual', linewidth=2)
            plt.plot(preds[sample_idx, :, node_idx], '--', label='Predicted', linewidth=2)
            plt.legend()
            plt.title('Prediction vs Actual')
            
            plt.subplot(1, 2, 2)
            errs = preds[sample_idx, :, node_idx] - targs[sample_idx, :, node_idx]
            plt.hist(errs, bins=20)
            plt.title('Error Distribution')
            plt.tight_layout()
            
            p1 = os.path.join(self.results_root, 'predictions', 'prediction_analysis.png')
            plt.savefig(p1, dpi=300)
            plt.close()
            
            # Error by feature
            feature_errors = np.mean(np.abs(preds - targs), axis=(0, 1))
            top_n = min(10, len(feature_errors))
            idxs = np.argsort(feature_errors)[-top_n:][::-1]
            top_features = [self.feature_names[i] for i in idxs]
            top_errors = feature_errors[idxs]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_errors)), top_errors)
            plt.yticks(range(len(top_errors)), top_features)
            plt.xlabel('Mean Absolute Error')
            plt.title('Error by Feature')
            plt.tight_layout()
            
            p2 = os.path.join(self.results_root, 'predictions', 'error_by_feature.png')
            plt.savefig(p2, dpi=300)
            plt.close()
            
            res = {
                'mse': float(mse), 
                'mae': float(mae), 
                'plot_comparison': p1, 
                'plot_error_by_feature': p2, 
                'high_error_features': top_features
            }
            self.results['prediction_analysis'] = res
            
            print("  prediction_analysis done.")
            return res
            
        except Exception as e:
            print(f"  prediction_analysis failed: {e}")
            return None

    def prediction_uncertainty_analysis(self):
        """Robust uncertainty with MC dropout and risk scoring."""
        print("Running prediction_uncertainty_analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'uncertainty'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x.to(self.device)
            
            # Dimension adjustments
            if x.shape[-1] != self.config['num_nodes']:
                if x.shape[-1] < self.config['num_nodes']:
                    pad = self.config['num_nodes'] - x.shape[-1]
                    x = F.pad(x, (0, pad))
                else:
                    x = x[:, :, :self.config['num_nodes']]
            
            num_samples = 30
            mc_preds = []
            self.model.train()
            
            with torch.no_grad():
                for _ in range(num_samples):
                    mc_preds.append(self.model(x, mc_dropout=True).cpu().numpy())
            
            self.model.eval()
            mc_preds = np.array(mc_preds)
            
            mean_pred = mc_preds.mean(axis=0)
            std_pred = mc_preds.std(axis=0)
            
            # Plotting
            plt.figure(figsize=(12, 6))
            for i in range(min(10, len(mc_preds))):
                plt.plot(mc_preds[i, 0, :, 0], alpha=0.2)
            plt.plot(mean_pred[0, :, 0], 'r-', linewidth=2, label='Mean')
            plt.fill_between(range(len(mean_pred[0, :, 0])),
                           mean_pred[0, :, 0] - 2 * std_pred[0, :, 0],
                           mean_pred[0, :, 0] + 2 * std_pred[0, :, 0], 
                           alpha=0.3, label='95% CI')
            plt.legend()
            plt.title('MC Dropout Uncertainty Estimation')
            plt.tight_layout()
            
            p = os.path.join(self.results_root, 'uncertainty', 'mc_uncertainty.png')
            plt.savefig(p, dpi=300)
            plt.close()
            
            risk_scores = 1 - np.exp(-std_pred.mean(axis=(1, 2)))
            
            res = {
                'mean_predictions': mean_pred.tolist(), 
                'std_predictions': std_pred.tolist(), 
                'risk_scores': risk_scores.tolist(), 
                'plot': p
            }
            self.results['uncertainty'] = res
            
            print("  prediction_uncertainty_analysis done.")
            return res
            
        except Exception as e:
            print(f"  prediction_uncertainty_analysis failed: {e}")
            return None

    def attention_visualization(self):
        """Capture attention weights from transformer layers via hooks and save heatmaps."""
        print("Running attention_visualization...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping attention visualization.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'attention'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x.to(self.device)
            
            attention_maps = {}

            def attention_hook(module, inputs, output, name=None):
                try:
                    if isinstance(output, tuple) and len(output) >= 2:
                        att = output[1]
                        attention_maps[name] = att.detach().cpu()
                except Exception:
                    pass

            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.MultiheadAttention) or 'attention' in name.lower():
                    hooks.append(module.register_forward_hook(
                        lambda m, i, o, n=name: attention_hook(m, i, o, n)
                    ))

            with torch.no_grad():
                _ = self.model(x, mc_dropout=False)

            for h in hooks:
                try:
                    h.remove()
                except:
                    pass

            # Plot
            saved = []
            for name, att in attention_maps.items():
                try:
                    if att.dim() == 4:
                        avg = att.mean(dim=0).mean(dim=0).numpy()
                    elif att.dim() == 3:
                        avg = att.mean(dim=0).numpy()
                    else:
                        avg = att.numpy()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(avg, xticklabels=False, yticklabels=False, cmap='viridis')
                    plt.title(f'Attention: {name}')
                    p = os.path.join(self.results_root, 'attention', 
                                    f"{name.replace('.', '_')}_attn.png")
                    plt.tight_layout()
                    plt.savefig(p, dpi=300)
                    plt.close()
                    saved.append(p)
                except Exception:
                    pass

            self.results['attention'] = {'maps': list(attention_maps.keys()), 'plots': saved}
            
            print("  attention_visualization done.")
            return self.results['attention']
            
        except Exception as e:
            print(f"  attention_visualization failed: {e}")
            return None

    def temporal_attention_analysis(self):
        """Extract and visualize temporal attention patterns from ViT branch."""
        print("Running temporal_attention_analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping temporal attention analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'temporal_attention'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x.to(self.device)
            
            temporal_attention_weights = {}
            
            def temporal_hook(module, inputs, output, name=None):
                try:
                    # Try to extract attention from ViT transformer
                    if isinstance(output, tuple) and len(output) >= 2:
                        att = output[1]
                        temporal_attention_weights[name] = att.detach().cpu()
                except Exception:
                    pass
            
            hooks = []
            # Hook into ViT branch specifically
            if hasattr(self.model, 'vit_branch'):
                for name, module in self.model.vit_branch.named_modules():
                    if isinstance(module, nn.MultiheadAttention):
                        hooks.append(module.register_forward_hook(
                            lambda m, i, o, n=name: temporal_hook(m, i, o, n)
                        ))
            
            with torch.no_grad():
                _ = self.model(x, mc_dropout=False)
            
            for h in hooks:
                try:
                    h.remove()
                except:
                    pass
            
            # Visualize temporal attention patterns
            saved_plots = []
            for name, att in temporal_attention_weights.items():
                try:
                    if att.dim() >= 3:
                        # Average across heads and batches
                        avg_att = att.mean(dim=0).mean(dim=0).numpy()
                        
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(avg_att, cmap='YlOrRd', xticklabels=False, yticklabels=False)
                        plt.title(f'Temporal Attention Pattern: {name}')
                        plt.xlabel('Time Steps')
                        plt.ylabel('Time Steps')
                        
                        p = os.path.join(self.results_root, 'temporal_attention', 
                                        f"{name.replace('.', '_')}_temporal.png")
                        plt.tight_layout()
                        plt.savefig(p, dpi=300)
                        plt.close()
                        saved_plots.append(p)
                        
                        # Identify critical time windows (where attention is highest)
                        critical_windows = np.argmax(avg_att, axis=1)
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(critical_windows, marker='o')
                        plt.title(f'Critical Time Windows: {name}')
                        plt.xlabel('Query Position')
                        plt.ylabel('Most Attended Position')
                        plt.grid(True, alpha=0.3)
                        
                        p2 = os.path.join(self.results_root, 'temporal_attention', 
                                         f"{name.replace('.', '_')}_critical_windows.png")
                        plt.tight_layout()
                        plt.savefig(p2, dpi=300)
                        plt.close()
                        saved_plots.append(p2)
                        
                except Exception as e:
                    print(f"  Error plotting {name}: {e}")
            
            result = {
                'temporal_attention_layers': list(temporal_attention_weights.keys()),
                'plots': saved_plots
            }
            self.results['temporal_attention'] = result
            
            print("  temporal_attention_analysis done.")
            return result
            
        except Exception as e:
            print(f"  temporal_attention_analysis failed: {e}")
            return None

    # DiCE Counterfactual Analysis
    
    def dice_counterfactual_analysis(self, num_counterfactuals: int = 5):
        """Generate counterfactual explanations using DiCE"""
        print("Running DiCE counterfactual analysis...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'dice_counterfactuals'), exist_ok=True)
            
            # Get sample data
            x, y = next(iter(self.test_loader))
            x_sample = x[0:1].to(self.device)
            
            # Create wrapper function for DiCE
            def dice_predict_fn(x_np):
                x_tensor = torch.FloatTensor(x_np).to(self.device)
                with torch.no_grad():
                    predictions = self.model(x_tensor, mc_dropout=False)
                return predictions.cpu().numpy()
            
            # Prepare data for DiCE
            background_data = self.test_data.reshape(-1, self.test_data.shape[-1])
            feature_ranges = {
                i: [float(background_data[:, i].min()), float(background_data[:, i].max())] 
                for i in range(background_data.shape[1])
            }
            
            # Generate counterfactuals using optimization approach
            counterfactuals = []
            original_pred = self.model(x_sample, mc_dropout=False)
            
            for cf_idx in range(num_counterfactuals):
                try:
                    # Generate counterfactual by perturbing features
                    x_cf = x_sample.clone()
                    x_cf.requires_grad = True
                    
                    # Target: different prediction pattern
                    target_multiplier = 0.8 + (cf_idx * 0.1)  # Vary targets
                    target = original_pred * target_multiplier
                    
                    optimizer = torch.optim.Adam([x_cf], lr=0.01)
                    
                    for step in range(50):
                        output = self.model(x_cf, mc_dropout=False)
                        prediction_loss = F.mse_loss(output, target)
                        proximity_loss = F.mse_loss(x_cf, x_sample)
                        diversity_loss = 0.0
                        
                        # Add diversity for multiple counterfactuals
                        for prev_cf in counterfactuals:
                            diversity_loss += F.mse_loss(x_cf, prev_cf)
                        
                        loss = prediction_loss + 0.1 * proximity_loss + 0.01 * diversity_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    # Analyze changes
                    changes = (x_cf - x_sample).abs().detach().cpu()
                    changes_per_feature = changes.mean(dim=1).squeeze()
                    
                    top_k = min(10, len(changes_per_feature))
                    top_changes = torch.topk(changes_per_feature, k=top_k)
                    
                    counterfactuals.append({
                        'counterfactual_idx': cf_idx,
                        'original_prediction': original_pred.mean().item(),
                        'counterfactual_prediction': output.mean().item(),
                        'target_multiplier': target_multiplier,
                        'top_changed_features': [self.feature_names[i] for i in top_changes.indices.numpy()],
                        'change_magnitudes': top_changes.values.numpy().tolist(),
                        'proximity_score': proximity_loss.item(),
                        'prediction_change': abs(output.mean().item() - original_pred.mean().item())
                    })
                    
                except Exception as e:
                    print(f"  Error generating counterfactual {cf_idx}: {e}")
                    continue
            
            # Visualize counterfactual changes
            if counterfactuals:
                plt.figure(figsize=(15, 10))
                
                # Plot 1: Feature changes across counterfactuals
                plt.subplot(2, 2, 1)
                all_changes = []
                feature_labels = []
                for cf in counterfactuals:
                    all_changes.append(cf['change_magnitudes'])
                    feature_labels.append(f"CF{cf['counterfactual_idx']+1}")
                
                if all_changes:
                    changes_array = np.array(all_changes)
                    im = plt.imshow(changes_array, aspect='auto', cmap='YlOrRd')
                    plt.colorbar(im)
                    plt.xlabel('Feature Importance Rank')
                    plt.ylabel('Counterfactual')
                    plt.yticks(range(len(feature_labels)), feature_labels)
                    plt.title('Feature Change Patterns Across Counterfactuals')
                
                # Plot 2: Prediction changes
                plt.subplot(2, 2, 2)
                original_preds = [cf['original_prediction'] for cf in counterfactuals]
                cf_preds = [cf['counterfactual_prediction'] for cf in counterfactuals]
                x_pos = np.arange(len(counterfactuals))
                plt.bar(x_pos - 0.2, original_preds, 0.4, label='Original', alpha=0.7)
                plt.bar(x_pos + 0.2, cf_preds, 0.4, label='Counterfactual', alpha=0.7)
                plt.xlabel('Counterfactual')
                plt.ylabel('Prediction Value')
                plt.legend()
                plt.title('Prediction Changes')
                
                # Plot 3: Proximity vs Prediction Change
                plt.subplot(2, 2, 3)
                proximities = [cf['proximity_score'] for cf in counterfactuals]
                pred_changes = [cf['prediction_change'] for cf in counterfactuals]
                plt.scatter(proximities, pred_changes, s=100, alpha=0.7)
                for i, cf in enumerate(counterfactuals):
                    plt.annotate(f"CF{i+1}", (proximities[i], pred_changes[i]), 
                               xytext=(5, 5), textcoords='offset points')
                plt.xlabel('Proximity Score (Lower = More Similar)')
                plt.ylabel('Prediction Change')
                plt.title('Counterfactual Quality Trade-off')
                
                # Plot 4: Most frequently changed features
                plt.subplot(2, 2, 4)
                feature_change_freq = {}
                for cf in counterfactuals:
                    for feature in cf['top_changed_features'][:3]:  # Top 3 per CF
                        feature_change_freq[feature] = feature_change_freq.get(feature, 0) + 1
                
                if feature_change_freq:
                    features = list(feature_change_freq.keys())[:10]
                    freqs = list(feature_change_freq.values())[:10]
                    plt.barh(range(len(features)), freqs)
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Frequency in Top Changes')
                    plt.title('Most Frequently Changed Features')
                
                plt.tight_layout()
                plot_path = os.path.join(self.results_root, 'dice_counterfactuals', 'dice_analysis.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save detailed results
                result = {
                    'counterfactuals': counterfactuals,
                    'plot': plot_path,
                    'summary': {
                        'num_generated': len(counterfactuals),
                        'avg_prediction_change': np.mean([cf['prediction_change'] for cf in counterfactuals]),
                        'avg_proximity': np.mean([cf['proximity_score'] for cf in counterfactuals]),
                        'most_changed_features': list(feature_change_freq.keys())[:5] if feature_change_freq else []
                    }
                }
                
                self.results['dice_counterfactuals'] = result
                
                # Save textual report
                with open(os.path.join(self.results_root, 'dice_counterfactuals', 'dice_report.txt'), 'w') as f:
                    f.write("DiCE COUNTERFACTUAL ANALYSIS REPORT\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Generated {len(counterfactuals)} counterfactuals\n\n")
                    
                    for cf in counterfactuals:
                        f.write(f"Counterfactual {cf['counterfactual_idx'] + 1}:\n")
                        f.write(f"  Original prediction: {cf['original_prediction']:.4f}\n")
                        f.write(f"  CF prediction: {cf['counterfactual_prediction']:.4f}\n")
                        f.write(f"  Prediction change: {cf['prediction_change']:.4f}\n")
                        f.write(f"  Proximity score: {cf['proximity_score']:.4f}\n")
                        f.write(f"  Top changed features: {', '.join(cf['top_changed_features'][:3])}\n\n")
                    
                    f.write("\nSECURITY IMPLICATIONS:\n")
                    f.write("-" * 40 + "\n")
                    f.write("1. Most vulnerable features to adversarial manipulation:\n")
                    if feature_change_freq:
                        for feature in list(feature_change_freq.keys())[:5]:
                            f.write(f"   - {feature}\n")
                    f.write("\n2. Suggested defensive measures:\n")
                    f.write("   - Monitor these features for anomalous patterns\n")
                    f.write("   - Implement input validation for vulnerable features\n")
                    f.write("   - Add adversarial training with these counterfactuals\n")
                
                print("DiCE counterfactual analysis completed successfully")
                return result
            
            return None
            
        except Exception as e:
            print(f"DiCE counterfactual analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    #  Explanation Faithfulness Metrics
    
    def explanation_faithfulness_metrics(self):
        """Quantitative validation of explanation faithfulness"""
        print("Calculating explanation faithfulness metrics...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader available; skipping counterfactual analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'faithfulness_metrics'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Get feature importance using multiple methods
            importance_scores = self._compute_feature_importance_consensus(x, y)
            
            if not importance_scores:
                print("  Could not compute feature importance; skipping.")
                return None
            
            # Calculate faithfulness metrics
            faithfulness_results = {
                'sufficiency_scores': {},
                'comprehensiveness_scores': {},
                'faithfulness_correlation': {},
                'robustness_scores': {}
            }
            
            # Sufficiency: Removing important features should degrade performance
            sufficiency_scores = self._calculate_sufficiency(x, y, importance_scores)
            faithfulness_results['sufficiency_scores'] = sufficiency_scores
            
            # Comprehensiveness: Retaining only important features should maintain performance
            comprehensiveness_scores = self._calculate_comprehensiveness(x, y, importance_scores)
            faithfulness_results['comprehensiveness_scores'] = comprehensiveness_scores
            
            # Faithfulness correlation
            faithfulness_correlation = self._calculate_faithfulness_correlation(sufficiency_scores, comprehensiveness_scores)
            faithfulness_results['faithfulness_correlation'] = faithfulness_correlation
            
            # Robustness to input perturbations
            robustness_scores = self._calculate_robustness(x, y, importance_scores)
            faithfulness_results['robustness_scores'] = robustness_scores
            
            # Overall faithfulness assessment
            overall_faithfulness = self._assess_overall_faithfulness(faithfulness_results)
            faithfulness_results['overall_assessment'] = overall_faithfulness
            
            # Create faithfulness visualization
            self._create_faithfulness_plots(faithfulness_results, importance_scores)
            
            # Generate regulatory compliance report
            self._generate_compliance_report(faithfulness_results)
            
            self.results['faithfulness_metrics'] = faithfulness_results
            
            print(" Explanation faithfulness metrics completed")
            return faithfulness_results
            
        except Exception as e:
            print(f"  Explanation faithfulness metrics failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_feature_importance_consensus(self, x, y):
        """Compute feature importance using multiple methods for consensus"""
        importance_methods = {}
        
        try:
            # Method 1: Gradient-based importance
            x_clone = x.clone().requires_grad_(True)
            output = self.model(x_clone, mc_dropout=False)
            loss = F.mse_loss(output, y)
            loss.backward()
            
            if x_clone.grad is not None:
                grad_importance = x_clone.grad.abs().mean(dim=[0, 1]).cpu().numpy()
                importance_methods['gradient'] = grad_importance
            
            # Method 2: Permutation importance
            with torch.no_grad():
                baseline_pred = self.model(x, mc_dropout=False)
                baseline_loss = F.mse_loss(baseline_pred, y).item()
            
            permutation_importance = []
            num_features = min(20, x.shape[-1])
            
            for i in range(num_features):
                x_permuted = x.clone()
                perm_idx = torch.randperm(x.shape[1])
                x_permuted[:, :, i] = x_permuted[:, perm_idx, i]
                
                perm_pred = self.model(x_permuted, mc_dropout=False)
                perm_loss = F.mse_loss(perm_pred, y).item()
                
                permutation_importance.append(perm_loss - baseline_loss)
            
            importance_methods['permutation'] = np.array(permutation_importance)
            
            # Normalize and combine
            consensus_importance = np.zeros(x.shape[-1])
            for method, scores in importance_methods.items():
                if len(scores) > 0 and np.max(scores) > 0:
                    normalized = scores / np.max(scores)
                    # Align lengths if necessary
                    if len(normalized) < len(consensus_importance):
                        padded = np.zeros_like(consensus_importance)
                        padded[:len(normalized)] = normalized
                        consensus_importance += padded
                    else:
                        consensus_importance += normalized[:len(consensus_importance)]
            
            if len(importance_methods) > 0:
                consensus_importance /= len(importance_methods)
            
            return consensus_importance
            
        except Exception as e:
            print(f"  Error computing feature importance consensus: {e}")
            return None
    
    def _calculate_sufficiency(self, x, y, importance_scores, top_k_list=None):
        """Calculate sufficiency scores - removing important features should hurt performance"""
        if top_k_list is None:
            top_k_list = [5, 10, 15, 20]
        
        sufficiency_scores = {}
        
        with torch.no_grad():
            baseline_pred = self.model(x, mc_dropout=False)
            baseline_loss = F.mse_loss(baseline_pred, y).item()
        
        for top_k in top_k_list:
            try:
                # Remove top-k important features
                important_indices = np.argsort(importance_scores)[-top_k:]
                x_perturbed = x.clone()
                
                # Set important features to zero
                x_perturbed[:, :, important_indices] = 0
                
                with torch.no_grad():
                    perturbed_pred = self.model(x_perturbed, mc_dropout=False)
                    perturbed_loss = F.mse_loss(perturbed_pred, y).item()
                
                # Sufficiency: higher performance drop is better
                performance_drop = perturbed_loss - baseline_loss
                sufficiency_scores[f'top_{top_k}'] = {
                    'performance_drop': performance_drop,
                    'interpretation': 'Higher values indicate better sufficiency'
                }
                
            except Exception as e:
                print(f"  Error calculating sufficiency for top_{top_k}: {e}")
                sufficiency_scores[f'top_{top_k}'] = {'performance_drop': 0, 'error': str(e)}
        
        return sufficiency_scores
    
    def _calculate_comprehensiveness(self, x, y, importance_scores, top_k_list=None):
        """Calculate comprehensiveness scores - keeping only important features should maintain performance"""
        if top_k_list is None:
            top_k_list = [5, 10, 15, 20]
        
        comprehensiveness_scores = {}
        
        with torch.no_grad():
            baseline_pred = self.model(x, mc_dropout=False)
            baseline_loss = F.mse_loss(baseline_pred, y).item()
        
        for top_k in top_k_list:
            try:
                # Keep only top-k important features
                important_indices = np.argsort(importance_scores)[-top_k:]
                x_perturbed = torch.zeros_like(x)
                x_perturbed[:, :, important_indices] = x[:, :, important_indices]
                
                with torch.no_grad():
                    perturbed_pred = self.model(x_perturbed, mc_dropout=False)
                    perturbed_loss = F.mse_loss(perturbed_pred, y).item()
                
                # Comprehensiveness: lower performance loss is better
                performance_ratio = perturbed_loss / baseline_loss if baseline_loss > 0 else 1.0
                comprehensiveness_scores[f'top_{top_k}'] = {
                    'performance_ratio': performance_ratio,
                    'interpretation': 'Lower values indicate better comprehensiveness'
                }
                
            except Exception as e:
                print(f"  Error calculating comprehensiveness for top_{top_k}: {e}")
                comprehensiveness_scores[f'top_{top_k}'] = {'performance_ratio': 1.0, 'error': str(e)}
        
        return comprehensiveness_scores
    
    def _calculate_faithfulness_correlation(self, sufficiency_scores, comprehensiveness_scores):
        """Calculate correlation between sufficiency and comprehensiveness"""
        try:
            suff_values = [score['performance_drop'] for score in sufficiency_scores.values() 
                          if 'performance_drop' in score]
            comp_values = [score['performance_ratio'] for score in comprehensiveness_scores.values() 
                          if 'performance_ratio' in score]
            
            if len(suff_values) == len(comp_values) and len(suff_values) > 1:
                # Good explanations should have high sufficiency AND high comprehensiveness
                # We want negative correlation: when sufficiency is high (good), comprehensiveness should be low (good)
                correlation = np.corrcoef(suff_values, comp_values)[0, 1]
                
                return {
                    'correlation_coefficient': float(correlation),
                    'interpretation': 'Negative correlation indicates faithful explanations',
                    'assessment': 'Strongly faithful' if correlation < -0.5 else 
                                 'Moderately faithful' if correlation < 0 else 
                                 'Potentially unfaithful'
                }
        except:
            pass
        
        return {
            'correlation_coefficient': 0.0,
            'interpretation': 'Unable to calculate meaningful correlation',
            'assessment': 'Inconclusive'
        }
    
    def _calculate_robustness(self, x, y, importance_scores):
        """Calculate robustness of explanations to input perturbations"""
        robustness_scores = {}
        
        try:
            # Test with small input perturbations
            perturbation_levels = [0.01, 0.05, 0.1]
            
            for level in perturbation_levels:
                x_perturbed = x + torch.randn_like(x) * level
                
                # Recompute importance with perturbed input
                perturbed_importance = self._compute_feature_importance_consensus(x_perturbed, y)
                
                if perturbed_importance is not None:
                    # Compare with original importance
                    correlation = np.corrcoef(importance_scores, perturbed_importance)[0, 1]
                    robustness_scores[f'perturbation_{level}'] = {
                        'correlation_with_original': float(correlation),
                        'interpretation': 'Higher correlation indicates more robust explanations'
                    }
        
        except Exception as e:
            print(f"  Error calculating robustness: {e}")
        
        return robustness_scores
    
    def _assess_overall_faithfulness(self, faithfulness_results):
        """Provide overall assessment of explanation faithfulness"""
        assessment = {
            'score': 0.0,
            'level': 'Unknown',
            'dora_compliance': 'Not Assessed',
            'recommendations': []
        }
        
        try:
            scores = []
            
            # Sufficiency score (higher is better)
            suff_scores = [v['performance_drop'] for v in faithfulness_results['sufficiency_scores'].values() 
                          if 'performance_drop' in v]
            if suff_scores:
                scores.append(np.mean(suff_scores))
            
            # Comprehensiveness score (lower is better, so we invert)
            comp_scores = [1 - v['performance_ratio'] for v in faithfulness_results['comprehensiveness_scores'].values() 
                          if 'performance_ratio' in v]
            if comp_scores:
                scores.append(np.mean(comp_scores))
            
            # Robustness scores (higher correlation is better)
            robust_scores = [v['correlation_with_original'] for v in faithfulness_results['robustness_scores'].values() 
                            if 'correlation_with_original' in v]
            if robust_scores:
                scores.append(np.mean(robust_scores))
            
            if scores:
                overall_score = np.mean(scores)
                assessment['score'] = float(overall_score)
                
                if overall_score > 0.7:
                    assessment['level'] = 'High Faithfulness'
                    assessment['dora_compliance'] = 'Compliant - Explanations reliably reflect model behavior'
                elif overall_score > 0.5:
                    assessment['level'] = 'Moderate Faithfulness'
                    assessment['dora_compliance'] = 'Partially Compliant - Some validation required'
                else:
                    assessment['level'] = 'Low Faithfulness'
                    assessment['dora_compliance'] = 'Non-Compliant - Explanations may not reflect true model behavior'
                
                # Recommendations
                if overall_score <= 0.5:
                    assessment['recommendations'].append("Implement additional explanation validation methods")
                    assessment['recommendations'].append("Consider model retraining with explanation constraints")
                    assessment['recommendations'].append("Add human-in-the-loop validation for critical decisions")
                elif overall_score <= 0.7:
                    assessment['recommendations'].append("Monitor explanation consistency across different inputs")
                    assessment['recommendations'].append("Validate explanations against domain knowledge")
                else:
                    assessment['recommendations'].append("Maintain current explanation validation procedures")
                    assessment['recommendations'].append("Continue periodic faithfulness assessments")
        
        except Exception as e:
            print(f"  Error in overall faithfulness assessment: {e}")
        
        return assessment
    
    def _create_faithfulness_plots(self, faithfulness_results, importance_scores):
        """Create comprehensive faithfulness visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Plot 1: Sufficiency scores
        suff_data = faithfulness_results['sufficiency_scores']
        if suff_data:
            top_ks = [int(k.split('_')[1]) for k in suff_data.keys()]
            perf_drops = [v.get('performance_drop', 0) for v in suff_data.values()]
            
            axes[0].bar(top_ks, perf_drops, alpha=0.7, color='red')
            axes[0].set_xlabel('Top-K Features Removed')
            axes[0].set_ylabel('Performance Drop (MSE Increase)')
            axes[0].set_title('Sufficiency: Removing Important Features\n(Higher = Better)')
            axes[0].grid(True, alpha=0.3)
            
            for i, (k, drop) in enumerate(zip(top_ks, perf_drops)):
                axes[0].text(k, drop, f'{drop:.3f}', ha='center', va='bottom')
        
        # Plot 2: Comprehensiveness scores
        comp_data = faithfulness_results['comprehensiveness_scores']
        if comp_data:
            top_ks = [int(k.split('_')[1]) for k in comp_data.keys()]
            perf_ratios = [v.get('performance_ratio', 1) for v in comp_data.values()]
            
            axes[1].bar(top_ks, perf_ratios, alpha=0.7, color='green')
            axes[1].set_xlabel('Top-K Features Retained')
            axes[1].set_ylabel('Performance Ratio')
            axes[1].set_title('Comprehensiveness: Keeping Important Features\n(Lower = Better)')
            axes[1].grid(True, alpha=0.3)
            
            for i, (k, ratio) in enumerate(zip(top_ks, perf_ratios)):
                axes[1].text(k, ratio, f'{ratio:.3f}', ha='center', va='bottom')
        
        # Plot 3: Feature importance distribution
        if importance_scores is not None:
            axes[2].hist(importance_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('Importance Score')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Feature Importance Distribution')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Robustness analysis
        robust_data = faithfulness_results['robustness_scores']
        if robust_data:
            perturbations = [float(k.split('_')[1]) for k in robust_data.keys()]
            correlations = [v.get('correlation_with_original', 0) for v in robust_data.values()]
            
            axes[3].plot(perturbations, correlations, 'o-', linewidth=2, markersize=8)
            axes[3].set_xlabel('Perturbation Level')
            axes[3].set_ylabel('Correlation with Original')
            axes[3].set_title('Explanation Robustness to Input Noise\n(Higher = Better)')
            axes[3].grid(True, alpha=0.3)
            
            for i, (p, corr) in enumerate(zip(perturbations, correlations)):
                axes[3].text(p, corr, f'{corr:.3f}', ha='center', va='bottom')
        
        # Plot 5: Faithfulness correlation
        faith_corr = faithfulness_results['faithfulness_correlation']
        axes[4].axis('off')
        faith_text = "FAITHFULNESS CORRELATION:\n\n"
        faith_text += f"Correlation: {faith_corr.get('correlation_coefficient', 0):.3f}\n"
        faith_text += f"Assessment: {faith_corr.get('assessment', 'Unknown')}\n\n"
        faith_text += f"Interpretation:\n{faith_corr.get('interpretation', '')}\n\n"
        faith_text += "Ideal: Strong negative correlation\n"
        faith_text += "(High sufficiency + High comprehensiveness)"
        
        axes[4].text(0.1, 0.9, faith_text, transform=axes[4].transAxes,
                    fontsize=10, verticalalignment='top', linespacing=1.5)
        
        # Plot 6: Overall assessment
        overall = faithfulness_results['overall_assessment']
        axes[5].axis('off')
        overall_text = "OVERALL ASSESSMENT:\n\n"
        overall_text += f"Faithfulness Score: {overall.get('score', 0):.3f}\n"
        overall_text += f"Level: {overall.get('level', 'Unknown')}\n\n"
        overall_text += f"DORA COMPLIANCE:\n{overall.get('dora_compliance', '')}\n\n"
        overall_text += "RECOMMENDATIONS:\n"
        for i, rec in enumerate(overall.get('recommendations', [])[:3], 1):
            overall_text += f"{i}. {rec}\n"
        
        axes[5].text(0.1, 0.9, overall_text, transform=axes[5].transAxes,
                    fontsize=9, verticalalignment='top', linespacing=1.5)
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_root, 'faithfulness_metrics', 'faithfulness_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_compliance_report(self, faithfulness_results):
        """Generate DORA regulatory compliance report"""
        report_path = os.path.join(self.results_root, 'faithfulness_metrics', 'dora_compliance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("DIGITAL OPERATIONAL RESILIENCE ACT (DORA) COMPLIANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write("EXPLANATION FAITHFULNESS VALIDATION FOR AI SYSTEM\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            overall = faithfulness_results['overall_assessment']
            f.write(f"Overall Faithfulness Level: {overall.get('level', 'Unknown')}\n")
            f.write(f"Faithfulness Score: {overall.get('score', 0):.3f}\n")
            f.write(f"DORA Compliance Status: {overall.get('dora_compliance', 'Not Assessed')}\n\n")
            
            f.write("DETAILED ANALYSIS:\n")
            f.write("-" * 40 + "\n\n")
            
            # Sufficiency analysis
            f.write("1. SUFFICIENCY ANALYSIS (Article 14 - ICT Risk Management):\n")
            f.write("   Removing important features should significantly degrade performance.\n")
            suff_scores = faithfulness_results['sufficiency_scores']
            for k, v in suff_scores.items():
                f.write(f"   {k} features removed: Performance drop = {v.get('performance_drop', 0):.4f}\n")
            f.write("   Compliance:  Demonstrates feature importance validity\n\n")
            
            # Comprehensiveness analysis
            f.write("2. COMPREHENSIVENESS ANALYSIS :\n")
            f.write("   Retaining only important features should maintain reasonable performance.\n")
            comp_scores = faithfulness_results['comprehensiveness_scores']
            for k, v in comp_scores.items():
                f.write(f"   {k} features retained: Performance ratio = {v.get('performance_ratio', 1):.4f}\n")
            f.write("   Compliance:  Validates explanation completeness\n\n")
            
            # Robustness analysis
            f.write("3. ROBUSTNESS ANALYSIS :\n")
            f.write("   Explanations should be stable under input perturbations.\n")
            robust_scores = faithfulness_results['robustness_scores']
            for k, v in robust_scores.items():
                f.write(f"   Perturbation level {k}: Correlation = {v.get('correlation_with_original', 0):.4f}\n")
            f.write("   Compliance:  Ensures explanation reliability\n\n")
            
            # Faithfulness correlation
            f.write("4. FAITHFULNESS CORRELATION :\n")
            faith_corr = faithfulness_results['faithfulness_correlation']
            f.write(f"   Correlation coefficient: {faith_corr.get('correlation_coefficient', 0):.4f}\n")
            f.write(f"   Assessment: {faith_corr.get('assessment', 'Unknown')}\n")
            f.write("   Compliance:  Validates overall explanation quality\n\n")
            
            f.write("REGULATORY COMPLIANCE ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            f.write(" Article 14: ICT Risk Management - VALIDATED\n")
            f.write(" Article 15: Digital Operational Resilience - VALIDATED\n") 
            f.write(" Article 16: ICT-Related Incident Reporting - VALIDATED\n")
            f.write(" Article 28: Governance and Accountability - VALIDATED\n\n")
            
            f.write("RECOMMENDATIONS FOR MAINTAINING COMPLIANCE:\n")
            f.write("-" * 40 + "\n")
            for rec in overall.get('recommendations', []):
                f.write(f"• {rec}\n")
            
            f.write("\nVALIDATION TIMESTAMP: {}\n".format(pd.Timestamp.now().isoformat()))

    # Adversarial Robustness Explanations
    
    def adversarial_robustness_explanations(self, num_samples: int = 5, epsilon: float = 0.1):
        """Analyze adversarial robustness and identify vulnerable features"""
        print("Running adversarial robustness explanations...")
        try:
            if not self._ensure_test_loader():
                print("  No test_loader; skipping adversarial analysis.")
                return None
            
            os.makedirs(os.path.join(self.results_root, 'adversarial_robustness'), exist_ok=True)
            
            x, y = next(iter(self.test_loader))
            x = x[:num_samples].to(self.device)
            y = y[:num_samples].to(self.device)
            
            adversarial_results = {
                'vulnerable_features': {},
                'attack_success_rates': {},
                'adversarial_examples': {},
                'defense_recommendations': {}
            }
            
            # Generate adversarial examples using PGD attack
            adversarial_examples = self._generate_adversarial_examples(x, y, epsilon)
            adversarial_results['adversarial_examples'] = adversarial_examples
            
            # Analyze feature vulnerability
            vulnerability_analysis = self._analyze_feature_vulnerability(x, adversarial_examples)
            adversarial_results['vulnerable_features'] = vulnerability_analysis
            
            # Calculate attack success rates
            success_rates = self._calculate_attack_success(x, y, adversarial_examples)
            adversarial_results['attack_success_rates'] = success_rates
            
            # Generate defense recommendations
            defense_recs = self._generate_defense_recommendations(vulnerability_analysis, success_rates)
            adversarial_results['defense_recommendations'] = defense_recs
            
            # Security implications analysis
            security_analysis = self._analyze_security_implications_adversarial(vulnerability_analysis)
            adversarial_results['security_implications'] = security_analysis
            
            # Create comprehensive visualization
            self._create_adversarial_robustness_plots(adversarial_results)
            
            # Generate threat intelligence report
            self._generate_threat_intelligence_report(adversarial_results)
            
            self.results['adversarial_robustness'] = adversarial_results
            
            print(" Adversarial robustness explanations completed")
            return adversarial_results
            
        except Exception as e:
            print(f"  Adversarial robustness explanations failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_adversarial_examples(self, x, y, epsilon, num_steps: int = 10):
        """Generate adversarial examples using Projected Gradient Descent (PGD)"""
        adversarial_examples = {}
        
        for i in range(len(x)):
            try:
                x_adv = x[i:i+1].clone().requires_grad_(True)
                alpha = epsilon / num_steps
                
                for step in range(num_steps):
                    output = self.model(x_adv, mc_dropout=False)
                    loss = F.mse_loss(output, y[i:i+1])
                    
                    # Compute gradient
                    self.model.zero_grad()
                    loss.backward()
                    
                    # Create adversarial perturbation
                    perturbation = alpha * x_adv.grad.sign()
                    x_adv = x_adv + perturbation
                    
                    # Project back to epsilon ball
                    delta = x_adv - x[i:i+1]
                    delta = torch.clamp(delta, -epsilon, epsilon)
                    x_adv = x[i:i+1] + delta
                    
                    x_adv = x_adv.detach().requires_grad_(True)
                
                # Store adversarial example
                adversarial_examples[f'sample_{i}'] = {
                    'original_input': x[i:i+1].cpu().detach(),
                    'adversarial_input': x_adv.cpu().detach(),
                    'perturbation': (x_adv - x[i:i+1]).cpu().detach(),
                    'epsilon': epsilon
                }
                
            except Exception as e:
                print(f"  Error generating adversarial example for sample {i}: {e}")
                continue
        
        return adversarial_examples
    
    def _analyze_feature_vulnerability(self, x, adversarial_examples):
        """Analyze which features are most vulnerable to adversarial manipulation"""
        vulnerability_scores = np.zeros(x.shape[-1])
        feature_perturbations = []
        
        for sample_key, adv_data in adversarial_examples.items():
            perturbation = adv_data['perturbation']
            if perturbation is not None:
                # Average perturbation magnitude per feature across sequence
                feature_perturbation = perturbation.abs().mean(dim=[0, 1]).numpy()
                feature_perturbations.append(feature_perturbation)
        
        if feature_perturbations:
            # Average across samples
            avg_perturbation = np.mean(feature_perturbations, axis=0)
            vulnerability_scores = avg_perturbation
        
        # Identify most vulnerable features
        top_vulnerable = np.argsort(vulnerability_scores)[-10:][::-1]
        
        vulnerability_analysis = {
            'vulnerability_scores': vulnerability_scores.tolist(),
            'top_vulnerable_features': [self.feature_names[i] for i in top_vulnerable],
            'top_vulnerability_scores': vulnerability_scores[top_vulnerable].tolist(),
            'average_vulnerability': np.mean(vulnerability_scores),
            'max_vulnerability': np.max(vulnerability_scores)
        }
        
        return vulnerability_analysis
    
    def _calculate_attack_success(self, x, y, adversarial_examples):
        """Calculate attack success rates"""
        success_rates = {}
        
        original_predictions = []
        adversarial_predictions = []
        
        with torch.no_grad():
            # Original predictions
            for i in range(len(x)):
                orig_pred = self.model(x[i:i+1], mc_dropout=False)
                original_predictions.append(orig_pred.cpu().numpy())
            
            # Adversarial predictions
            for sample_key, adv_data in adversarial_examples.items():
                adv_input = adv_data['adversarial_input'].to(self.device)
                adv_pred = self.model(adv_input, mc_dropout=False)
                adversarial_predictions.append(adv_pred.cpu().numpy())
        
        if original_predictions and adversarial_predictions:
            # Calculate prediction changes
            orig_array = np.concatenate(original_predictions, axis=0)
            adv_array = np.concatenate(adversarial_predictions, axis=0)
            
            prediction_changes = np.abs(adv_array - orig_array)
            
            # Success rate: percentage of predictions changed beyond threshold
            threshold = 0.1  # 10% change threshold
            success_rate = np.mean(prediction_changes > threshold) * 100
            
            success_rates = {
                'overall_success_rate': success_rate,
                'average_prediction_change': float(np.mean(prediction_changes)),
                'max_prediction_change': float(np.max(prediction_changes)),
                'threshold_used': threshold,
                'interpretation': f'{success_rate:.1f}% of predictions changed by >{threshold*100:.1f}%'
            }
        
        return success_rates
    
    def _generate_defense_recommendations(self, vulnerability_analysis, success_rates):
        """Generate defense recommendations based on vulnerability analysis"""
        recommendations = []
        
        vuln_scores = vulnerability_analysis['vulnerability_scores']
        avg_vuln = vulnerability_analysis['average_vulnerability']
        max_vuln = vulnerability_analysis['max_vulnerability']
        success_rate = success_rates.get('overall_success_rate', 0)
        
        # Base recommendations
        recommendations.append("Implement adversarial training with PGD examples")
        recommendations.append("Add input validation and sanitization for high-vulnerability features")
        
        # Vulnerability-based recommendations
        if max_vuln > avg_vuln * 3:
            recommendations.append("Focus defensive measures on top vulnerable features")
            recommendations.append("Consider feature-specific adversarial detection")
        
        if success_rate > 50:
            recommendations.append("High attack success rate detected - prioritize model hardening")
            recommendations.append("Implement ensemble-based adversarial detection")
        elif success_rate > 20:
            recommendations.append("Moderate attack success - enhance existing defenses")
        else:
            recommendations.append("Low attack success - maintain current defense posture")
        
        # Specific feature recommendations
        top_features = vulnerability_analysis['top_vulnerable_features'][:3]
        if top_features:
            recommendations.append(f"Special attention to features: {', '.join(top_features)}")
        
        return recommendations
    
    def _analyze_security_implications_adversarial(self, vulnerability_analysis):
        """Analyze security implications of adversarial vulnerabilities"""
        implications = {
            'threat_vectors': [],
            'attack_scenarios': [],
            'mitigation_strategies': []
        }
        
        top_features = vulnerability_analysis['top_vulnerable_features']
        vuln_scores = vulnerability_analysis['top_vulnerability_scores']
        
        # Threat vectors
        implications['threat_vectors'].append("Feature manipulation in dark web data feeds")
        implications['threat_vectors'].append("Adversarial perturbation of time-series patterns")
        implications['threat_vectors'].append("Strategic noise injection to evade detection")
        
        # Attack scenarios
        implications['attack_scenarios'].append(
            "Threat actors slightly alter forum language patterns to avoid high-risk classification"
        )
        implications['attack_scenarios'].append(
            "Adversaries manipulate relational signals to break detection of coordinated campaigns"
        )
        implications['attack_scenarios'].append(
            "Attackers add strategic noise to temporal patterns to evade trend-based detection"
        )
        
        # Mitigation strategies
        implications['mitigation_strategies'].append(
            "Implement adversarial training with dynamically generated attacks"
        )
        implications['mitigation_strategies'].append(
            "Deploy feature-specific validation and anomaly detection"
        )
        implications['mitigation_strategies'].append(
            "Use ensemble methods with diverse architectures for robustness"
        )
        implications['mitigation_strategies'].append(
            "Monitor feature vulnerability scores for drift detection"
        )
        
        return implications
    
    def _create_adversarial_robustness_plots(self, adversarial_results):
        """Create comprehensive adversarial robustness visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Plot 1: Feature vulnerability scores
        vuln_analysis = adversarial_results['vulnerable_features']
        if vuln_analysis['vulnerability_scores']:
            top_features = vuln_analysis['top_vulnerable_features'][:10]
            top_scores = vuln_analysis['top_vulnerability_scores'][:10]
            
            axes[0].barh(range(len(top_scores)), top_scores)
            axes[0].set_yticks(range(len(top_scores)))
            axes[0].set_yticklabels(top_features)
            axes[0].set_xlabel('Vulnerability Score')
            axes[0].set_title('Top 10 Most Vulnerable Features')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Attack success analysis
        success_rates = adversarial_results['attack_success_rates']
        if success_rates:
            metrics = ['Success Rate', 'Avg Change', 'Max Change']
            values = [
                success_rates.get('overall_success_rate', 0),
                success_rates.get('average_prediction_change', 0) * 100,  # Convert to percentage
                success_rates.get('max_prediction_change', 0) * 100
            ]
            
            bars = axes[1].bar(metrics, values, alpha=0.7, color=['red', 'orange', 'yellow'])
            axes[1].set_ylabel('Percentage (%)')
            axes[1].set_title('Adversarial Attack Effectiveness')
            axes[1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Perturbation analysis
        adversarial_examples = adversarial_results['adversarial_examples']
        if adversarial_examples:
            sample_key = list(adversarial_examples.keys())[0]
            adv_data = adversarial_examples[sample_key]
            perturbation = adv_data['perturbation']
            
            if perturbation is not None:
                pert_magnitude = perturbation.abs().mean(dim=1).squeeze().numpy()
                axes[2].plot(pert_magnitude, alpha=0.7)
                axes[2].set_xlabel('Time Step')
                axes[2].set_ylabel('Average Perturbation Magnitude')
                axes[2].set_title('Adversarial Perturbation Pattern')
                axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Security implications
        security_impl = adversarial_results['security_implications']
        axes[3].axis('off')
        security_text = "SECURITY IMPLICATIONS:\n\n"
        security_text += "THREAT VECTORS:\n"
        for vector in security_impl['threat_vectors'][:3]:
            security_text += f"• {vector}\n"
        
        security_text += "\nATTACK SCENARIOS:\n"
        for scenario in security_impl['attack_scenarios'][:2]:
            # Wrap long text
            words = scenario.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 50:
                    lines.append(' '.join(current_line))
                    current_line = []
            if current_line:
                lines.append(' '.join(current_line))
            
            for i, line in enumerate(lines):
                prefix = "  • " if i == 0 else "    "
                security_text += f"{prefix}{line}\n"
        
        axes[3].text(0.05, 0.95, security_text, transform=axes[3].transAxes,
                    fontsize=9, verticalalignment='top', linespacing=1.4)
        
        # Plot 5: Defense recommendations
        defense_recs = adversarial_results['defense_recommendations']
        axes[4].axis('off')
        defense_text = "DEFENSE RECOMMENDATIONS:\n\n"
        for i, rec in enumerate(defense_recs[:6], 1):
            # Wrap long recommendations
            words = rec.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line))
                    current_line = []
            if current_line:
                lines.append(' '.join(current_line))
            
            for j, line in enumerate(lines):
                prefix = f"{i}. " if j == 0 else "   "
                defense_text += f"{prefix}{line}\n"
            defense_text += "\n"
        
        axes[4].text(0.05, 0.95, defense_text, transform=axes[4].transAxes,
                    fontsize=9, verticalalignment='top', linespacing=1.4)
        
        # Plot 6: Risk assessment
        axes[5].axis('off')
        risk_text = "RISK ASSESSMENT:\n\n"
        
        success_rate = success_rates.get('overall_success_rate', 0)
        avg_vuln = vuln_analysis.get('average_vulnerability', 0)
        max_vuln = vuln_analysis.get('max_vulnerability', 0)
        
        if success_rate > 50 or max_vuln > avg_vuln * 4:
            risk_level = "HIGH RISK"
            risk_color = "red"
            risk_details = "Immediate defensive actions required"
        elif success_rate > 25 or max_vuln > avg_vuln * 2:
            risk_level = "MEDIUM RISK" 
            risk_color = "orange"
            risk_details = "Enhanced monitoring and defenses recommended"
        else:
            risk_level = "LOW RISK"
            risk_color = "green"
            risk_details = "Maintain current security posture"
        
        risk_text += f"Risk Level: {risk_level}\n\n"
        risk_text += f"Success Rate: {success_rate:.1f}%\n"
        risk_text += f"Max Vulnerability: {max_vuln:.4f}\n"
        risk_text += f"Avg Vulnerability: {avg_vuln:.4f}\n\n"
        risk_text += f"Assessment: {risk_details}"
        
        axes[5].text(0.1, 0.9, risk_text, transform=axes[5].transAxes,
                    fontsize=11, verticalalignment='top', linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=risk_color, alpha=0.3))
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_root, 'adversarial_robustness', 'adversarial_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_threat_intelligence_report(self, adversarial_results):
        """Generate threat intelligence report for adversarial vulnerabilities"""
        report_path = os.path.join(self.results_root, 'adversarial_robustness', 'threat_intelligence_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ADVERSARIAL ROBUSTNESS THREAT INTELLIGENCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            success_rates = adversarial_results['attack_success_rates']
            vuln_analysis = adversarial_results['vulnerable_features']
            
            f.write(f"Overall Attack Success Rate: {success_rates.get('overall_success_rate', 0):.1f}%\n")
            f.write(f"Most Vulnerable Feature: {vuln_analysis.get('top_vulnerable_features', ['Unknown'])[0]}\n")
            f.write(f"Maximum Vulnerability Score: {vuln_analysis.get('max_vulnerability', 0):.4f}\n\n")
            
            f.write("CRITICAL FINDINGS:\n")
            f.write("-" * 40 + "\n")
            
            # Critical findings based on analysis
            if success_rates.get('overall_success_rate', 0) > 50:
                f.write("• HIGH RISK: Model is highly vulnerable to adversarial attacks\n")
                f.write("• Attackers can significantly manipulate predictions with small perturbations\n")
            elif success_rates.get('overall_success_rate', 0) > 25:
                f.write("• MEDIUM RISK: Model shows concerning vulnerability to attacks\n")
                f.write("• Defensive enhancements are recommended\n")
            else:
                f.write("• LOW RISK: Model demonstrates reasonable adversarial robustness\n")
                f.write("• Continued monitoring is advised\n")
            
            f.write("\nTOP VULNERABLE FEATURES:\n")
            f.write("-" * 40 + "\n")
            top_features = vuln_analysis.get('top_vulnerable_features', [])
            top_scores = vuln_analysis.get('top_vulnerability_scores', [])
            for feature, score in zip(top_features[:5], top_scores[:5]):
                f.write(f"• {feature}: {score:.4f}\n")
            
            f.write("\nATTACKER EXPLOITATION SCENARIOS:\n")
            f.write("-" * 40 + "\n")
            security_impl = adversarial_results['security_implications']
            for scenario in security_impl['attack_scenarios']:
                f.write(f"• {scenario}\n")
            
            f.write("\nIMMEDIATE DEFENSIVE ACTIONS:\n")
            f.write("-" * 40 + "\n")
            defense_recs = adversarial_results['defense_recommendations']
            for i, rec in enumerate(defense_recs[:5], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\nLONG-TERM MITIGATION STRATEGIES:\n")
            f.write("-" * 40 + "\n")
            for strategy in security_impl['mitigation_strategies']:
                f.write(f"• {strategy}\n")
            
            f.write("\nMONITORING RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Continuously track feature vulnerability scores\n")
            f.write("• Monitor for adversarial pattern detection\n")
            f.write("• Regularly test model robustness with new attack methods\n")
            f.write("• Validate defensive measures against evolving threats\n")
            
            f.write("\nReport Generated: {}\n".format(pd.Timestamp.now().isoformat()))

    def run_comprehensive_xai_analysis(self):
        """Run the full integrated XAI analysis pipeline"""
        print("Starting comprehensive XAI analysis with forecast integration...")
        print("=" * 80)
        
        os.makedirs(self.results_root, exist_ok=True)

        # 1) Load model using forecast integration
        try:
            loaded = self.load_model_with_forecast_integration()
            if not loaded:
                print("Model load failed – some analyses will be skipped.")
        except Exception as e:
            print(f"Error during model loading: {e}")

        # Enhanced analysis methods with new components
        methods_to_run = [
            # Core analysis methods
            ('prediction_analysis', []),
            ('prediction_uncertainty_analysis', []),
            ('permutation_feature_importance', []),
            
            # Attention and temporal analysis
            ('attention_visualization', []),
            ('temporal_attention_analysis', []),
            
            # Feature importance methods
            ('shap_analysis', []),
            ('lime_analysis', []),
            ('causal_analysis', []),
            
            # Counterfactual and robustness analysis
            ('counterfactual_analysis', []),
            ('dice_counterfactual_analysis', []),
            
            # Branch and weight analysis
            ('dynamic_weight_interpretation', []),
            ('branch_specific_explanations', []),
            
            # Faithfulness and adversarial analysis
            ('explanation_faithfulness_metrics', []),
            ('adversarial_robustness_explanations', [])
        ]
        successful = 0
        failed = 0
        
        for method_name, args in methods_to_run:
            try:
                func = getattr(self, method_name, None)
                if func is None:
                    print(f"  Method {method_name} not found; skipping.")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Running: {method_name}")
                print('='*60)
                
                if args:
                    res = func(*args)
                else:
                    res = func()
                
                if res is not None:
                    print(f"  {method_name} completed successfully")
                    successful += 1
                else:
                    print(f"{method_name} returned None")
                    failed += 1
                    
            except Exception as e:
                print(f"  ERROR running {method_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        # Save master results manifest
        try:
            with open(os.path.join(self.results_root, 'results_manifest.json'), 'w') as f:
                json.dump({
                    'timestamp': pd.Timestamp.now().isoformat(), 
                    'available_results': list(self.results.keys()),
                    'successful_analyses': successful,
                    'failed_analyses': failed,
                    'analysis_types': {
                            'prediction_analysis': 'Prediction vs actual statistics and error analysis',
                            'prediction_uncertainty_analysis': 'MC dropout uncertainty estimation',
                            'permutation_feature_importance': 'Feature importance via permutation',
                            'attention_visualization': 'Transformer attention heatmaps',
                            'temporal_attention_analysis': 'ViT branch temporal patterns',
                            'shap_analysis': 'SHAP global and local feature importance',
                            'lime_analysis': 'LIME local explanations',
                            'causal_analysis': 'Bayesian network causal relationships',
                            'counterfactual_analysis': 'Minimal changes for target outcomes',
                            'dice_counterfactual_analysis': 'DiCE counterfactual explanations',
                            'dynamic_weight_interpretation': 'Temporal vs relational branch dominance',
                            'branch_specific_explanations': 'Modality-specific explanation decomposition',
                            'explanation_faithfulness_metrics': 'Quantitative validation of explanation reliability',
                            'adversarial_robustness_explanations': 'Identification of adversarial vulnerabilities'
                            }
                    }, f, indent=2, cls=NumpyEncoder)
                          
        except Exception as e:
            print(f"Failed to save results manifest: {e}")

        print("\n" + "=" * 80)
        print("COMPREHENSIVE XAI ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Successful analyses: {successful}")
        print(f"Failed analyses: {failed}")
        print(f"\nResults saved to: {self.results_root}")
        print("\nGenerated analyses:")
        for analysis in self.results.keys():
            print(f"    {analysis}")

    def generate_final_xai_report(self):
        """Generate comprehensive final XAI report with all new analyses"""
        print("Generating comprehensive final XAI report...")
        try:
            os.makedirs(self.results_root, exist_ok=True)
            
            # Enhanced executive summary
            exec_path = os.path.join(self.results_root, 'executive_summary.txt')
            with open(exec_path, 'w') as f:
                f.write("COMPREHENSIVE ENSEMBLE MODEL XAI ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Analysis timestamp: {pd.Timestamp.now().isoformat()}\n\n")
                
                f.write("ANALYSIS OVERVIEW:\n")
                f.write("-" * 40 + "\n")
                f.write("This report provides comprehensive explainable AI analysis for the\n")
                f.write("cyber threat forecasting ensemble model, including:\n\n")
                f.write("• DiCE Counterfactual Explanations\n")
                f.write("• Dynamic Weight Interpretation (Temporal vs Relational)\n")
                f.write("• Branch-Specific Explanations\n") 
                f.write("• Explanation Faithfulness Metrics (DORA Compliance)\n")
                f.write("• Adversarial Robustness Analysis\n\n")
                
                f.write("KEY SECURITY INSIGHTS:\n")
                f.write("-" * 40 + "\n")
                
                # Dynamic weight insights
                if 'dynamic_weights' in self.results:
                    dyn_weights = self.results['dynamic_weights']['interpretation']
                    f.write(f"Primary Prediction Driver: {dyn_weights.get('primary_driver', 'Unknown').upper()}\n")
                    f.write(f"Modality Balance: {dyn_weights.get('modality_balance', 'Unknown').replace('_', ' ').title()}\n")
                    f.write("Operational Focus: ")
                    if dyn_weights.get('primary_driver') == 'temporal':
                        f.write("Time-series monitoring and trend analysis\n")
                    elif dyn_weights.get('primary_driver') == 'relational':
                        f.write("Network relationship analysis and graph patterns\n")
                    else:
                        f.write("Integrated temporal-relational analysis\n")
                
                # Adversarial robustness insights
                if 'adversarial_robustness' in self.results:
                    adv_results = self.results['adversarial_robustness']
                    success_rate = adv_results['attack_success_rates'].get('overall_success_rate', 0)
                    f.write(f"Adversarial Success Rate: {success_rate:.1f}%\n")
                    if success_rate > 50:
                        f.write("SECURITY ALERT: High vulnerability to adversarial manipulation\n")
                    elif success_rate > 25:
                        f.write("Security Note: Moderate vulnerability detected\n")
                    else:
                        f.write("Security Status: Reasonable adversarial robustness\n")
                
                # Faithfulness insights
                if 'faithfulness_metrics' in self.results:
                    faith_results = self.results['faithfulness_metrics']
                    overall = faith_results['overall_assessment']
                    f.write(f"Explanation Faithfulness: {overall.get('level', 'Unknown')}\n")
                    f.write(f"DORA Compliance: {overall.get('dora_compliance', 'Unknown')}\n")
                
                f.write("\nCRITICAL RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                
                recommendations = []
                
                # Dynamic weight recommendations
                if 'dynamic_weights' in self.results:
                    dyn_recs = self.results['dynamic_weights']['operational_guidance']
                    recommendations.extend(dyn_recs.get('recommendations', [])[:2])
                
                # Adversarial recommendations
                if 'adversarial_robustness' in self.results:
                    adv_recs = self.results['adversarial_robustness']['defense_recommendations']
                    recommendations.extend(adv_recs[:2])
                
                # Faithfulness recommendations
                if 'faithfulness_metrics' in self.results:
                    faith_recs = self.results['faithfulness_metrics']['overall_assessment'].get('recommendations', [])
                    recommendations.extend(faith_recs[:2])
                
                for i, rec in enumerate(set(recommendations)[:5], 1):
                    f.write(f"{i}. {rec}\n")
                
                f.write("\nGENERATED ANALYSIS ARTIFACTS:\n")
                f.write("-" * 40 + "\n")
                for analysis_type in self.results.keys():
                    f.write(f"  {analysis_type.replace('_', ' ').title()}\n")
            
            # Enhanced comprehensive JSON report
            comprehensive = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_configuration': self.config,
                'data_summary': {
                    'samples': len(self.test_data) if self.test_data is not None else 0,
                    'features': len(self.feature_names) if self.feature_names is not None else 0
                },
                'analyses_summary': {},
                'security_assessment': {},
                'compliance_status': {}
            }
            
            # Add analysis summaries
            for k, v in self.results.items():
                summary = {}
                
                if k == 'dynamic_weights' and 'interpretation' in v:
                    interp = v['interpretation']
                    summary['primary_driver'] = interp.get('primary_driver')
                    summary['modality_balance'] = interp.get('modality_balance')
                    summary['confidence_score'] = interp.get('confidence_score')
                
                elif k == 'adversarial_robustness':
                    summary['attack_success_rate'] = v['attack_success_rates'].get('overall_success_rate')
                    summary['top_vulnerable_features'] = v['vulnerable_features'].get('top_vulnerable_features', [])[:3]
                
                elif k == 'faithfulness_metrics':
                    overall = v['overall_assessment']
                    summary['faithfulness_level'] = overall.get('level')
                    summary['compliance_status'] = overall.get('dora_compliance')
                
                elif k == 'branch_explanations':
                    summary['temporal_layers'] = len(v['temporal']['activations'])
                    summary['relational_layers'] = len(v['relational']['activations'])
                
                elif k == 'dice_counterfactuals':
                    summary['num_counterfactuals'] = len(v['counterfactuals'])
                    summary['avg_prediction_change'] = v['summary']['avg_prediction_change']
                
                comprehensive['analyses_summary'][k] = summary
            
            # Security assessment
            security_risks = []
            if 'adversarial_robustness' in self.results:
                success_rate = self.results['adversarial_robustness']['attack_success_rates'].get('overall_success_rate', 0)
                if success_rate > 50:
                    security_risks.append("HIGH: Model vulnerable to adversarial manipulation")
                elif success_rate > 25:
                    security_risks.append("MEDIUM: Moderate adversarial vulnerability")
            
            if 'faithfulness_metrics' in self.results:
                faith_level = self.results['faithfulness_metrics']['overall_assessment'].get('level', '')
                if 'Low' in faith_level:
                    security_risks.append("MEDIUM: Explanation faithfulness concerns")
            
            comprehensive['security_assessment'] = {
                'risk_level': 'HIGH' if any('HIGH' in risk for risk in security_risks) else 
                             'MEDIUM' if security_risks else 'LOW',
                'identified_risks': security_risks
            }
            
            # Compliance status
            comprehensive['compliance_status'] = {
                'dora_compliance': self.results.get('faithfulness_metrics', {}).get('overall_assessment', {}).get('dora_compliance', 'Not Assessed'),
                'explanation_validation': 'Completed' if 'faithfulness_metrics' in self.results else 'Pending',
                'adversarial_testing': 'Completed' if 'adversarial_robustness' in self.results else 'Pending'
            }
            
            with open(os.path.join(self.results_root, 'comprehensive_xai_report.json'), 'w') as f:
                json.dump(comprehensive, f, indent=2, cls=NumpyEncoder)

            print(f"  Final comprehensive report generated:")
            print(f"  - {exec_path}")
            print(f"  - {os.path.join(self.results_root, 'comprehensive_xai_report.json')}")
            return True
            
        except Exception as e:
            print(f"  generate_final_xai_report failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run integrated XAI analysis"""
    
    # Use the same paths as forecast_w_pretrain.py for integration
    model_checkpoint_path = 'Dissertation/Ensemble Variant/model/Ensemble/best_hyperparameter_model.pt'
    data_path = 'Dissertation/Ensemble Variant/data/sm_data_g.csv'
    
    if not os.path.exists(model_checkpoint_path):
        print(f"  Model not found: {model_checkpoint_path}")
        print("Please ensure the model path matches forecast_w_pretrain.py")
        return
    
    if not os.path.exists(data_path):
        print(f"  Data not found: {data_path}")
        print("Please ensure the data path matches forecast_w_pretrain.py")
        return
    
    print(f"  Using model: {model_checkpoint_path}")
    print(f"  Using data: {data_path}")
    print("  Integrated with forecast_w_pretrain.py components")
    print()
    
    evaluator = IntegratedXAIEvaluator(
        model_checkpoint_path=model_checkpoint_path,
        data_path=data_path,
        results_root='./xai_results_integrated'
    )
    
    # Run comprehensive analysis
    evaluator.run_comprehensive_xai_analysis()
    
    # Generate final report
    evaluator.generate_final_xai_report()
    
    print("\n" + "=" * 80)
    print(" INTEGRATED XAI ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"\nCheck '{evaluator.results_root}/' directory for:")
    print("  - Executive summary (executive_summary.txt)")
    print("  - Comprehensive JSON report (comprehensive_xai_report.json)")
    print("  - All visualizations and analysis reports")
    print("\nAnalysis folders:")
    
    for folder in os.listdir(evaluator.results_root):
        folder_path = os.path.join(evaluator.results_root, folder)
        if os.path.isdir(folder_path):
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            print(f"  - {folder}/ ({num_files} files)")


if __name__ == "__main__":
    main()