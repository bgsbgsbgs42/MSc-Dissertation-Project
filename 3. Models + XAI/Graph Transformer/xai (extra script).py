from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import os
import sys
import csv
import json
from collections import defaultdict
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GNNExplainer at Dynamic Graph Construction Phase

class ThreatGraphExplainer:
    """
    Explains dynamic threat correlation graphs using GNNExplainer
    Identifies critical edges and nodes influencing predictions
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device

        class _GraphModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x, edge_index, edge_attr=None, pe=None, batch=None):
                if edge_attr is None:
                    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
                if pe is None:
                    raise ValueError("Positional encodings (pe) are required for GraphTransformer.")
                if batch is None:
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                return self.base_model(x, pe, edge_index, edge_attr, batch)

        # Initialize GNNExplainer
        self.explainer = Explainer(
            model=_GraphModelWrapper(model),
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )
    
    def explain_threat_prediction(
        self, 
        graph_data, 
        node_idx: int,
        threat_name: str = "Unknown Threat"
    ) -> Dict:
        """
        Generate explanation for a specific threat node prediction
        
        Args:
            graph_data: PyG Data object with threat correlation graph
            node_idx: Index of threat category node to explain
            threat_name: Name of threat category
            
        Returns:
            Dictionary containing explanation artifacts
        """
        self.model.eval()
        
        # Generate explanation
        batch = getattr(graph_data, 'batch', torch.zeros(graph_data.x.size(0), dtype=torch.long, device=graph_data.x.device))
        explanation = self.explainer(
            graph_data.x,
            graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            pe=graph_data.pe,
            batch=batch
        )
        
        # Extract edge and node masks
        edge_mask = explanation.edge_mask.cpu().numpy()
        node_mask = explanation.node_mask.cpu().numpy() if hasattr(explanation, 'node_mask') else None
        
        # Get top-k most influential edges
        k = min(10, len(edge_mask))
        top_edge_indices = np.argsort(edge_mask)[-k:][::-1]
        
        # Extract influential threat relationships
        edge_index = graph_data.edge_index.cpu().numpy()
        influential_edges = []
        
        for idx in top_edge_indices:
            src, dst = edge_index[0, idx], edge_index[1, idx]
            weight = edge_mask[idx]
            correlation = graph_data.edge_attr[idx].item() if graph_data.edge_attr is not None else 0.0
            
            influential_edges.append({
                'source_threat': src,
                'target_threat': dst,
                'importance_score': float(weight),
                'temporal_correlation': float(correlation),
                'source_name': graph_data.threat_names[src] if hasattr(graph_data, 'threat_names') else f"Threat_{src}",
                'target_name': graph_data.threat_names[dst] if hasattr(graph_data, 'threat_names') else f"Threat_{dst}"
            })
        
        # Compute subgraph fidelity
        fidelity = self._compute_fidelity(graph_data, node_idx, explanation)
        
        return {
            'threat_name': threat_name,
            'node_idx': node_idx,
            'influential_edges': influential_edges,
            'edge_mask': edge_mask,
            'node_mask': node_mask,
            'fidelity_score': fidelity,
            'explanation_object': explanation
        }
    
    def _compute_fidelity(self, graph_data, node_idx, explanation) -> float:
        """
        Compute fidelity: prediction change when important edges removed
        """
        # Original prediction
        batch = getattr(graph_data, 'batch', torch.zeros(graph_data.x.size(0), dtype=torch.long, device=graph_data.x.device))
        with torch.no_grad():
            original_pred = self.model(graph_data.x, graph_data.pe, graph_data.edge_index, graph_data.edge_attr, batch)[node_idx]
        
        # Create masked graph (remove top edges)
        edge_mask = explanation.edge_mask
        threshold = torch.quantile(edge_mask, 0.9)  # Keep only top 10%
        kept_mask = edge_mask < threshold
        masked_edge_index = graph_data.edge_index[:, kept_mask]
        masked_edge_attr = graph_data.edge_attr[kept_mask] if graph_data.edge_attr is not None else None
        with torch.no_grad():
            masked_pred = self.model(graph_data.x, graph_data.pe, masked_edge_index, masked_edge_attr, batch)[node_idx]
        
        # Fidelity: relative change in prediction
        fidelity = torch.abs(original_pred - masked_pred) / (torch.abs(original_pred) + 1e-8)
        
        return fidelity.item()
    
    def visualize_explanatory_subgraph(
        self, 
        explanation_dict: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize the explanatory subgraph with threat relationships
        """
        import networkx as nx
        
        edges = explanation_dict['influential_edges']
        
        # Create directed graph
        G = nx.DiGraph()
        
        for edge in edges:
            G.add_edge(
                edge['source_name'],
                edge['target_name'],
                weight=edge['importance_score'],
                correlation=edge['temporal_correlation']
            )
        
        # Plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue',
            node_size=3000,
            alpha=0.9
        )
        
        # Draw edges with varying thickness based on importance
        edges_list = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges_list]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 5 for w in weights],
            alpha=0.6,
            edge_color=weights,
            edge_cmap=plt.cm.Reds,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add edge labels with correlation values
        edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in edges_list}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title(f"Explanatory Subgraph for {explanation_dict['threat_name']}\nFidelity Score: {explanation_dict['fidelity_score']:.3f}", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Attention Weight Analysis at GPSConv Dual-Pathway Processing

class GPSConvAttentionAnalyzer:
    """
    Analyzes and visualizes attention patterns in GPS Transformer layers
    Separates local message passing and global attention pathways
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {
            'local': [],
            'global': []
        }
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        
        def local_attention_hook(module, input, output):
            # Capture local message passing attention
            if hasattr(module, 'attention_weights_local'):
                self.attention_weights['local'].append(
                    module.attention_weights_local.detach().cpu()
                )
        
        def global_attention_hook(module, input, output):
            # Capture global transformer attention
            if hasattr(module, 'attention_weights_global'):
                self.attention_weights['global'].append(
                    module.attention_weights_global.detach().cpu()
                )
        
        # Register hooks on GPSConv layers
        for name, module in self.model.named_modules():
            if 'gpsconv' in name.lower() or 'gps' in name.lower():
                module.register_forward_hook(local_attention_hook)
                module.register_forward_hook(global_attention_hook)
    
    def analyze_forward_pass(
        self, 
        graph_data,
        target_node_idx: int
    ) -> Dict:
        """
        Run forward pass and collect attention weights
        """
        self.attention_weights = {'local': [], 'global': []}
        
        batch = getattr(graph_data, 'batch', torch.zeros(graph_data.x.size(0), dtype=torch.long, device=graph_data.x.device))
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                graph_data.x,
                graph_data.pe,
                graph_data.edge_index,
                graph_data.edge_attr,
                batch
            )
        
        # Aggregate attention across layers
        local_attention = self._aggregate_attention(self.attention_weights['local'], target_node_idx)
        global_attention = self._aggregate_attention(self.attention_weights['global'], target_node_idx)
        
        return {
            'local_attention': local_attention,
            'global_attention': global_attention,
            'prediction': output[target_node_idx].cpu().numpy()
        }
    
    def _aggregate_attention(
        self, 
        attention_list: List[torch.Tensor],
        target_node_idx: int
    ) -> np.ndarray:
        """
        Aggregate attention weights across layers for target node
        """
        if not attention_list:
            return np.array([])
        
        # Average attention across layers
        aggregated = torch.stack(attention_list).mean(dim=0)
        
        # Extract attention for target node
        if aggregated.dim() == 3:  # [num_heads, num_nodes, num_nodes]
            node_attention = aggregated[:, target_node_idx, :].mean(dim=0)
        elif aggregated.dim() == 2:  # [num_nodes, num_nodes]
            node_attention = aggregated[target_node_idx, :]
        else:
            node_attention = aggregated
        
        return node_attention.numpy()
    
    def create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        threat_names: List[str],
        pathway_type: str = "Local",
        target_threat: str = "Target Threat",
        save_path: Optional[str] = None
    ):
        """
        Create attention heatmap showing threat-to-threat attention patterns
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention_weights.reshape(1, -1) if attention_weights.ndim == 1 else attention_weights,
            xticklabels=threat_names if len(threat_names) == len(attention_weights) else range(len(attention_weights)),
            yticklabels=[target_threat],
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Attention Weight'},
            vmin=0,
            vmax=attention_weights.max()
        )
        
        plt.title(f'{pathway_type} Pathway Attention for {target_threat}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Source Threat Categories', fontsize=12)
        plt.ylabel('Target Threat', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def rank_influential_neighbors(
        self,
        local_attention: np.ndarray,
        graph_data,
        target_node_idx: int,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Rank most influential neighbor nodes based on local attention
        """
        # Get neighbor indices
        edge_index = graph_data.edge_index.cpu().numpy()
        neighbors = edge_index[1, edge_index[0] == target_node_idx]
        
        if len(neighbors) == 0:
            return pd.DataFrame()
        
        # Get attention scores for neighbors
        neighbor_attention = local_attention[neighbors]
        
        # Create ranking dataframe
        ranking_df = pd.DataFrame({
            'Neighbor_Node_Idx': neighbors,
            'Threat_Name': [graph_data.threat_names[n] if hasattr(graph_data, 'threat_names') else f"Threat_{n}" 
                           for n in neighbors],
            'Attention_Score': neighbor_attention,
            'Edge_Weight': [graph_data.edge_attr[i].item() if graph_data.edge_attr is not None else 0.0 
                           for i in range(len(graph_data.edge_attr)) 
                           if edge_index[0, i] == target_node_idx and edge_index[1, i] in neighbors][:len(neighbors)]
        })
        
        # Sort by attention score
        ranking_df = ranking_df.sort_values('Attention_Score', ascending=False).head(top_k)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df[['Rank', 'Threat_Name', 'Attention_Score', 'Edge_Weight']]
    
    def compute_pathway_contributions(
        self,
        local_attention: np.ndarray,
        global_attention: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute relative contributions of local vs global pathways
        """
        local_magnitude = np.sum(np.abs(local_attention))
        global_magnitude = np.sum(np.abs(global_attention))
        total = local_magnitude + global_magnitude
        
        return {
            'local_contribution_pct': (local_magnitude / total * 100) if total > 0 else 0,
            'global_contribution_pct': (global_magnitude / total * 100) if total > 0 else 0,
            'local_magnitude': local_magnitude,
            'global_magnitude': global_magnitude
        }


# Transformer Attention Rollout for Global Attention Flow

class TransformerAttentionRollout:
    """
    Implements attention rollout to trace global attention flow
    through multiple transformer layers
    """
    
    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.attention_matrices = []
    
    def compute_attention_rollout(
        self,
        graph_data,
        target_node_idx: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute attention rollout: accumulated attention from input to output
        
        Returns:
            rollout_attention: Final accumulated attention [num_nodes]
            layer_attentions: Attention at each layer
        """
        self.model.eval()
        
        # Collect attention weights from all layers
        layer_attentions = self._collect_layer_attentions(graph_data)
        
        if not layer_attentions:
            return np.array([]), []
        
        # Initialize rollout with identity matrix
        num_nodes = graph_data.num_nodes
        rollout = np.eye(num_nodes)
        
        # Accumulate attention through layers
        for layer_attn in layer_attentions:
            # Average over attention heads if multi-head
            if layer_attn.ndim == 3:  # [num_heads, num_nodes, num_nodes]
                layer_attn = layer_attn.mean(axis=0)
            
            # Add residual connection (50% attention, 50% identity)
            layer_attn = 0.5 * layer_attn + 0.5 * np.eye(num_nodes)
            
            # Matrix multiplication to accumulate attention
            rollout = np.matmul(layer_attn, rollout)
        
        # Extract attention flow to target node
        rollout_attention = rollout[target_node_idx, :]
        
        # Normalize
        rollout_attention = rollout_attention / (rollout_attention.sum() + 1e-8)
        
        return rollout_attention, layer_attentions
    
    def _collect_layer_attentions(self, graph_data) -> List[np.ndarray]:
        """
        Collect attention weights from all transformer layers
        """
        attentions = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                attentions.append(module.attention_weights.detach().cpu().numpy())
        
        # Register hooks
        hooks = []
        for module in self.model.modules():
            if hasattr(module, 'attention_weights'):
                hooks.append(module.register_forward_hook(attention_hook))
        
        # Forward pass
        with torch.no_grad():
            self.model(graph_data.x, graph_data.edge_index)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attentions
    
    def visualize_global_attention_flow(
        self,
        rollout_attention: np.ndarray,
        graph_data,
        target_node_idx: int,
        threshold: float = 0.05,
        save_path: Optional[str] = None
    ):
        """
        Visualize global attention flow as directed graph
        Shows which nodes have strongest influence through attention
        """
        import networkx as nx
        
        # Filter nodes by attention threshold
        significant_nodes = np.where(rollout_attention > threshold)[0]
        
        # Create directed graph
        G = nx.DiGraph()
        
        target_name = (graph_data.threat_names[target_node_idx] 
                      if hasattr(graph_data, 'threat_names') 
                      else f"Target_{target_node_idx}")
        
        # Add edges from significant nodes to target
        for node_idx in significant_nodes:
            if node_idx != target_node_idx:
                source_name = (graph_data.threat_names[node_idx] 
                             if hasattr(graph_data, 'threat_names') 
                             else f"Threat_{node_idx}")
                
                G.add_edge(
                    source_name,
                    target_name,
                    weight=rollout_attention[node_idx]
                )
        
        # Visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Node sizes based on attention weight
        node_sizes = []
        for node in G.nodes():
            if node == target_name:
                node_sizes.append(5000)
            else:
                # Find corresponding attention weight
                node_idx = [i for i, name in enumerate(graph_data.threat_names) 
                           if name == node][0] if hasattr(graph_data, 'threat_names') else 0
                node_sizes.append(rollout_attention[node_idx] * 10000)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=['red' if node == target_name else 'lightblue' for node in G.nodes()],
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 10 for w in weights],
            alpha=0.5,
            edge_color=weights,
            edge_cmap=plt.cm.Reds,
            arrows=True,
            arrowsize=25,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f'Global Attention Flow to {target_name}\n(Attention Rollout Visualization)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_attention_flow_matrix(
        self,
        layer_attentions: List[np.ndarray],
        threat_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Create layer-wise attention flow heatmap
        """
        num_layers = len(layer_attentions)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 6))
        
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_attn, ax) in enumerate(zip(layer_attentions, axes)):
            # Average over heads if necessary
            if layer_attn.ndim == 3:
                layer_attn = layer_attn.mean(axis=0)
            
            sns.heatmap(
                layer_attn,
                xticklabels=threat_names if len(threat_names) == layer_attn.shape[1] else [],
                yticklabels=threat_names if len(threat_names) == layer_attn.shape[0] else [],
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Attention Weight'},
                square=True
            )
            
            ax.set_title(f'Layer {idx+1} Attention', fontsize=12, fontweight='bold')
            ax.set_xlabel('Source Threats', fontsize=10)
            ax.set_ylabel('Target Threats', fontsize=10)
            
            if len(threat_names) == layer_attn.shape[1]:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# RWPE Ablation Studies - Positional Encoding Impact

class RWPEAblationAnalyzer:
    """
    Analyzes impact of Random Walk Positional Encoding through ablation
    Measures how structural position affects predictions
    """
    
    def __init__(self, model_with_pe, model_without_pe):
        """
        Args:
            model_with_pe: GPS model with RWPE enabled
            model_without_pe: GPS model with RWPE disabled
        """
        self.model_with_pe = model_with_pe
        self.model_without_pe = model_without_pe
    
    def compute_positional_importance(
        self,
        graph_data,
        num_iterations: int = 10
    ) -> Dict:
        """
        Compute importance of positional encoding for each node
        Uses prediction variance across models
        """
        self.model_with_pe.eval()
        self.model_without_pe.eval()
        
        predictions_with_pe = []
        predictions_without_pe = []
        
        # Multiple forward passes for robust estimation
        for _ in range(num_iterations):
            with torch.no_grad():
                pred_with = self.model_with_pe(graph_data.x, graph_data.edge_index)
                pred_without = self.model_without_pe(graph_data.x, graph_data.edge_index)
                
                predictions_with_pe.append(pred_with.cpu().numpy())
                predictions_without_pe.append(pred_without.cpu().numpy())
        
        # Average predictions
        avg_pred_with = np.mean(predictions_with_pe, axis=0)
        avg_pred_without = np.mean(predictions_without_pe, axis=0)
        
        # Compute importance: absolute difference in predictions
        positional_importance = np.abs(avg_pred_with - avg_pred_without)
        
        # Relative importance (normalized)
        relative_importance = positional_importance / (np.abs(avg_pred_with) + 1e-8)
        
        return {
            'absolute_importance': positional_importance,
            'relative_importance': relative_importance,
            'predictions_with_pe': avg_pred_with,
            'predictions_without_pe': avg_pred_without
        }
    
    def create_structural_importance_map(
        self,
        graph_data,
        importance_scores: np.ndarray,
        metric_name: str = "Positional Importance",
        save_path: Optional[str] = None
    ):
        """
        Visualize structural importance on graph layout
        """
        import networkx as nx
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        # Create NetworkX graph
        edge_index = graph_data.edge_index.cpu().numpy()
        G = nx.DiGraph()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Normalize importance scores for coloring
        norm = Normalize(vmin=importance_scores.min(), vmax=importance_scores.max())
        cmap = plt.cm.YlOrRd
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Draw nodes with color based on importance
        node_colors = [importance_scores[node] for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=800,
            cmap=cmap,
            vmin=importance_scores.min(),
            vmax=importance_scores.max(),
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.3,
            arrows=True,
            arrowsize=15,
            width=1.5
        )
        
        # Labels
        labels = {i: graph_data.threat_names[i] if hasattr(graph_data, 'threat_names') else f"T{i}" 
                 for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.set_label(metric_name, fontsize=12)
        
        plt.title(f'Structural Importance Map\n({metric_name} from RWPE Ablation)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_positional_clusters(
        self,
        graph_data,
        importance_scores: np.ndarray,
        n_clusters: int = 3
    ) -> pd.DataFrame:
        """
        Cluster threats by positional importance
        Identifies structural roles (e.g., entry-point, pivot, exfiltration)
        """
        from sklearn.cluster import KMeans
        
        # Reshape for clustering
        X = importance_scores.reshape(-1, 1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create analysis dataframe
        df = pd.DataFrame({
            'Threat_Idx': range(len(importance_scores)),
            'Threat_Name': [graph_data.threat_names[i] if hasattr(graph_data, 'threat_names') 
                           else f"Threat_{i}" for i in range(len(importance_scores))],
            'Positional_Importance': importance_scores,
            'Structural_Cluster': cluster_labels
        })
        
        # Sort by importance
        df = df.sort_values('Positional_Importance', ascending=False)
        
        # Add cluster descriptions based on importance level
        cluster_means = df.groupby('Structural_Cluster')['Positional_Importance'].mean().sort_values(ascending=False)
        cluster_map = {
            cluster_means.index[0]: 'High Positional Dependency',
            cluster_means.index[1]: 'Medium Positional Dependency' if n_clusters > 2 else 'Low Positional Dependency',
            cluster_means.index[2]: 'Low Positional Dependency' if n_clusters > 2 else None
        }
        
        df['Cluster_Description'] = df['Structural_Cluster'].map(cluster_map)
        
        return df
    
    def compute_performance_metrics(
        self,
        graph_data,
        ground_truth: np.ndarray
    ) -> Dict:
        """
        Compare model performance with/without RWPE
        """
        self.model_with_pe.eval()
        self.model_without_pe.eval()
        
        with torch.no_grad():
            pred_with_pe = self.model_with_pe(graph_data.x, graph_data.edge_index).cpu().numpy()
            pred_without_pe = self.model_without_pe(graph_data.x, graph_data.edge_index).cpu().numpy()
        
        # Calculate metrics
        mse_with = mean_squared_error(ground_truth, pred_with_pe)
        mse_without = mean_squared_error(ground_truth, pred_without_pe)
        
        mae_with = np.mean(np.abs(ground_truth - pred_with_pe))
        mae_without = np.mean(np.abs(ground_truth - pred_without_pe))
        
        improvement_mse = ((mse_without - mse_with) / mse_without) * 100
        improvement_mae = ((mae_without - mae_with) / mae_without) * 100
        
        return {
            'mse_with_pe': mse_with,
            'mse_without_pe': mse_without,
            'mae_with_pe': mae_with,
            'mae_without_pe': mae_without,
            'mse_improvement_pct': improvement_mse,
            'mae_improvement_pct': improvement_mae
        }


#  Bayesian Uncertainty Explainability at MC Dropout Layer

class BayesianUncertaintyAnalyzer:
    """
    Analyzes and decomposes uncertainty from MC Dropout
    Provides epistemic uncertainty quantification and attribution
    """
    
    def __init__(self, model, num_samples: int = 100):
        """
        Args:
            model: GPS model with MC Dropout layers
            num_samples: Number of MC samples for uncertainty estimation
        """
        self.model = model
        self.num_samples = num_samples
    
    def enable_dropout(self):
        """Enable dropout during inference for MC sampling"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout for deterministic inference"""
        self.model.eval()
    
    def compute_predictive_uncertainty(
        self,
        graph_data,
        target_nodes: Optional[List[int]] = None
    ) -> Dict:
        """
        Compute epistemic uncertainty using MC Dropout
        
        Returns:
            Dictionary with mean predictions, uncertainty estimates, and samples
        """
        self.model.eval()
        self.enable_dropout()  # Keep dropout active
        
        predictions = []
        
        # MC sampling
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(graph_data.x, graph_data.edge_index)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # [num_samples, num_nodes, num_features]
        
        # Compute statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Compute confidence intervals (95%)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        # Compute coefficient of variation (normalized uncertainty)
        cv = std_pred / (np.abs(mean_pred) + 1e-8)
        
        # Filter for target nodes if specified
        if target_nodes is not None:
            mean_pred = mean_pred[target_nodes]
            std_pred = std_pred[target_nodes]
            lower_bound = lower_bound[target_nodes]
            upper_bound = upper_bound[target_nodes]
            cv = cv[target_nodes]
            predictions = predictions[:, target_nodes, :]
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'lower_bound_95': lower_bound,
            'upper_bound_95': upper_bound,
            'coefficient_variation': cv,
            'all_samples': predictions
        }
    
    def decompose_uncertainty_sources(
        self,
        graph_data,
        node_idx: int,
        ablation_configs: Dict[str, any]
    ) -> Dict:
        """
        Decompose total uncertainty into component sources through ablation
        
        Args:
            graph_data: Input graph
            node_idx: Target node index
            ablation_configs: Dict with keys 'full_model', 'no_dwi', 'no_temporal', etc.
        
        Returns:
            Uncertainty decomposition by source
        """
        uncertainty_components = {}
        
        for config_name, model_variant in ablation_configs.items():
            # Compute uncertainty for this model variant
            self.model = model_variant
            result = self.compute_predictive_uncertainty(graph_data, target_nodes=[node_idx])
            
            uncertainty_components[config_name] = {
                'mean': result['mean_prediction'][0],
                'std': result['std_prediction'][0],
                'cv': result['coefficient_variation'][0]
            }
        
        # Compute attributions (how much each component contributes)
        full_uncertainty = uncertainty_components['full_model']['std']
        
        attributions = {}
        for config_name, metrics in uncertainty_components.items():
            if config_name != 'full_model':
                # Difference in uncertainty when component removed
                uncertainty_reduction = full_uncertainty - metrics['std']
                attributions[config_name] = {
                    'uncertainty_contribution': uncertainty_reduction,
                    'contribution_pct': (uncertainty_reduction / full_uncertainty * 100) if full_uncertainty > 0 else 0
                }
        
        return {
            'total_uncertainty': full_uncertainty,
            'uncertainty_components': uncertainty_components,
            'attributions': attributions
        }
    
    def create_uncertainty_breakdown_chart(
        self,
        decomposition: Dict,
        threat_name: str = "Target Threat",
        save_path: Optional[str] = None
    ):
        """
        Visualize uncertainty decomposition as stacked bar chart
        """
        attributions = decomposition['attributions']
        
        # Prepare data
        components = list(attributions.keys())
        contributions = [attributions[comp]['contribution_pct'] for comp in components]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart of contributions
        colors = plt.cm.Set3(range(len(components)))
        bars = ax1.barh(components, contributions, color=colors, alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Contribution to Total Uncertainty (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Uncertainty Source', fontsize=12, fontweight='bold')
        ax1.set_title(f'Uncertainty Decomposition for {threat_name}', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, contrib in zip(bars, contributions):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{contrib:.1f}%', va='center', fontweight='bold')
        
        # Pie chart
        ax2.pie(contributions, labels=components, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title('Relative Uncertainty Contributions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_uncertainty_timeline(
        self,
        predictions_over_time: List[Dict],
        threat_name: str = "Threat Category",
        time_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize predictions with uncertainty intervals over time horizon
        
        Args:
            predictions_over_time: List of uncertainty dicts for each time step
            threat_name: Name of threat being visualized
            time_labels: Labels for time axis (e.g., ["T+1", "T+2", ...])
        """
        # Extract data
        time_steps = len(predictions_over_time)
        means = [pred['mean_prediction'] for pred in predictions_over_time]
        stds = [pred['std_prediction'] for pred in predictions_over_time]
        lower = [pred['lower_bound_95'] for pred in predictions_over_time]
        upper = [pred['upper_bound_95'] for pred in predictions_over_time]
        
        if time_labels is None:
            time_labels = [f"T+{i+1}" for i in range(time_steps)]
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        x = np.arange(time_steps)
        
        # Plot mean prediction
        plt.plot(x, means, 'b-', linewidth=2.5, label='Mean Prediction', marker='o', markersize=8)
        
        # Plot confidence intervals
        plt.fill_between(x, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
        
        # Plot 1-std bands
        plt.fill_between(x, 
                        np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds),
                        alpha=0.5, color='lightblue', label='±1 Std Dev')
        
        # Formatting
        plt.xlabel('Forecast Horizon', fontsize=12, fontweight='bold')
        plt.ylabel('Threat Severity Score', fontsize=12, fontweight='bold')
        plt.title(f'Uncertainty-Aware Forecast for {threat_name}\n(36-Month Horizon with Epistemic Uncertainty)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x, time_labels, rotation=45, ha='right')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_uncertainty_confidence_matrix( self, graph_data, confidence_threshold: float = 0.8 ) -> pd.DataFrame:
        """
        Create matrix showing prediction confidence for all threats
        Used for risk-aware decision triage
        """
        # Compute uncertainty for all nodes
        uncertainty_result = self.compute_predictive_uncertainty(graph_data)
        
        mean_preds = uncertainty_result['mean_prediction'].flatten()
        std_preds = uncertainty_result['std_prediction'].flatten()
        cv_preds = uncertainty_result['coefficient_variation'].flatten()
        
        # Compute confidence scores (inverse of CV)
        confidence_scores = 1 / (1 + cv_preds)
        
        # Create dataframe
        df = pd.DataFrame({
            'Threat_Idx': range(len(mean_preds)),
            'Threat_Name': [graph_data.threat_names[i] if hasattr(graph_data, 'threat_names') 
                           else f"Threat_{i}" for i in range(len(mean_preds))],
            'Mean_Prediction': mean_preds,
            'Std_Deviation': std_preds,
            'Coefficient_Variation': cv_preds,
            'Confidence_Score': confidence_scores
        })
        
        # Add risk classification
        df['Confidence_Level'] = pd.cut(
            df['Confidence_Score'],
            bins=[0, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add decision recommendation
        def get_recommendation(row):
            if row['Confidence_Score'] >= confidence_threshold and row['Mean_Prediction'] > 7:
                return 'Automated Defense Activation'
            elif row['Confidence_Score'] >= 0.6 and row['Mean_Prediction'] > 7:
                return 'Human Review Required'
            elif row['Confidence_Score'] < 0.6:
                return 'Expert Analysis Needed'
            else:
                return 'Continue Monitoring'
        
        df['Action_Recommendation'] = df.apply(get_recommendation, axis=1)
        
        # Sort by combination of threat severity and confidence
        df['Priority_Score'] = df['Mean_Prediction'] * df['Confidence_Score']
        df = df.sort_values('Priority_Score', ascending=False)
        
        return df
    
    def create_confidence_heatmap(
        self,
        confidence_matrix: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Visualize confidence levels across all threats
        """
        # Prepare data for heatmap
        pivot_data = confidence_matrix.pivot_table(
            index='Threat_Name',
            values=['Mean_Prediction', 'Confidence_Score'],
            aggfunc='mean'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Heatmap 1: Threat Severity
        sns.heatmap(
            pivot_data[['Mean_Prediction']],
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Threat Severity'},
            ax=ax1,
            vmin=0,
            vmax=10
        )
        ax1.set_title('Mean Threat Predictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Threat Categories', fontsize=12)
        
        # Heatmap 2: Confidence Scores
        sns.heatmap(
            pivot_data[['Confidence_Score']],
            cmap='RdYlGn',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Confidence Score'},
            ax=ax2,
            vmin=0,
            vmax=1
        )
        ax2.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Define the GPS model architecture

class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pred_len = config['forecast_horizon']
        self.channels = config['channels']
        self.num_nodes = config['num_nodes']
        
        self.node_emb = nn.Linear(config['node_dim'], config['channels'] - config['pe_dim'])
        self.pe_lin = nn.Linear(20, config['pe_dim'])
        self.pe_norm = nn.BatchNorm1d(20)
        
        self.edge_emb = nn.Embedding(2, config['channels'])
        
        self.convs = nn.ModuleList()
        for _ in range(config['num_layers']):
            nn_seq = nn.Sequential(
                nn.Linear(config['channels'], config['channels']),
                nn.ReLU(),
                nn.Linear(config['channels'], config['channels']),
            )
            conv = GPSConv(config['channels'], GINEConv(nn_seq), 
                          heads=config.get('num_heads', 4),
                          attn_type=config['attn_type'], 
                          attn_kwargs={'dropout': config['dropout']})
            self.convs.append(conv)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.forecast_head = nn.Sequential(
            nn.Linear(config['channels'], config['channels'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['channels'] // 2, config['channels'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['channels'] // 4, config['forecast_horizon'] * config['num_nodes']),
        )
    
    def forward(self, x, pe, edge_index, edge_attr, batch=None):
        x_pe = self.pe_norm(pe)
        
        node_emb = self.node_emb(x)
        pe_emb = self.pe_lin(x_pe)
        
        x = torch.cat((node_emb, pe_emb), dim=1)
        edge_attr = self.edge_emb(edge_attr)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = self.dropout(x)
        
        x = global_add_pool(x, batch)
        return self.forecast_head(x)

# Data Loading and Preparation

def load_data_and_graph(data_file: str, graph_file: str):
    """Load data and graph for XAI analysis"""
    
    # Load data
    print(f"Loading data from {data_file}...")
    data_df = pd.read_csv(data_file)
    
    # Extract data values (skip date column)
    if 'Date' in data_df.columns[0]:
        dates = data_df.iloc[:, 0].tolist()
        data_values = data_df.iloc[:, 1:].values.astype(np.float32)
        column_names = data_df.columns[1:].tolist()
    else:
        dates = None
        data_values = data_df.iloc[:, :].values.astype(np.float32)
        column_names = data_df.columns.tolist()
    
    print(f"Data shape: {data_values.shape}")
    print(f"Number of threat categories: {len(column_names)}")
    
    # Load graph structure
    print(f"Loading graph from {graph_file}...")
    graph = defaultdict(list)
    
    with open(graph_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                key_node = row[0]
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
    
    print(f"Graph loaded with {len(graph)} attacks")
    
    return {
        'data': data_values,
        'column_names': column_names,
        'dates': dates,
        'graph': graph,
        'num_nodes': len(column_names)
    }

def create_correlation_graph(data: np.ndarray, column_names: List[str], threshold: float = 0.3):
    """Create graph structure based on correlation"""
    
    # Clean data
    data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute correlation
    correlation_matrix = np.corrcoef(data_clean.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    # Create adjacency matrix
    adj_matrix = (np.abs(correlation_matrix) > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
    
    # Create edge index and attributes
    edge_index = []
    edge_attr = []
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                edge_index.append([i, j])
                # Use correlation sign as edge attribute (1 for positive, 0 for negative)
                edge_attr.append(1 if correlation_matrix[i, j] > 0 else 0)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    print(f"Graph created: {edge_index.shape[1]} edges")
    
    return edge_index, edge_attr, correlation_matrix

def prepare_pyg_data(data: np.ndarray, edge_index: torch.Tensor, 
                     edge_attr: torch.Tensor, column_names: List[str]):
    """Prepare PyTorch Geometric data object"""
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
    
    # Use the most recent time step as node features
    x = torch.FloatTensor(data[-1]).unsqueeze(1)  # [num_nodes, 1]
    
    # Add positional encoding
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    
    data_obj = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(column_names)
    )
    
    # Store threat names for visualization
    data_obj.threat_names = column_names
    
    data_obj = transform(data_obj)
    
    return data_obj

def load_or_create_model(config: Dict, model_path: Optional[str] = None):
    """Load existing model or create new one for XAI"""
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_data = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in model_data:
            state_dict = model_data['model_state_dict']
            hp = model_data.get('hyperparameters', {})
        else:
            state_dict = model_data
            hp = {}
        
        # Update config with loaded hyperparameters
        config.update(hp)
        
        model = GraphTransformer(config).to(device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    else:
        print("Creating new model for XAI analysis...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GraphTransformer(config).to(device)
    
    return model

# XAI Visualization Pipeline

class XAIVisualizationPipeline:
    """Main pipeline for XAI visualizations"""
    
    def __init__(self, data_dir: str = 'data', output_dir: str = 'xai_visualizations'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'attention'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'uncertainty'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ablation'), exist_ok=True)
        
        # Configuration
        self.config = {
            'channels': 64,
            'num_layers': 3,
            'num_heads': 4,
            'pe_dim': 8,
            'node_dim': 1,
            'attn_type': 'multihead',
            'dropout': 0.1,
            'forecast_horizon': 36,
            'num_nodes': None  # Will be set dynamically
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.graph_data = None
        self.correlation_matrix = None
        self.column_names = None
    
    def load_and_prepare_data(self, data_file: str = 'sm_data_g.csv', 
                             graph_file: str = 'graph.csv'):
        """Load and prepare data for XAI analysis"""
        
        # Load data
        data_dict = load_data_and_graph(
            os.path.join(self.data_dir, data_file),
            os.path.join(self.data_dir, graph_file)
        )
        
        self.column_names = data_dict['column_names']
        self.config['num_nodes'] = len(self.column_names)
        
        # Create correlation graph
        edge_index, edge_attr, corr_matrix = create_correlation_graph(
            data_dict['data'], self.column_names, threshold=0.3
        )
        
        self.correlation_matrix = corr_matrix
        
        # Prepare PyG data
        self.graph_data = prepare_pyg_data(
            data_dict['data'], edge_index, edge_attr, self.column_names
        )
        
        # Move to device
        self.graph_data = self.graph_data.to(self.device)
        
        print(f"Data prepared: {self.graph_data.num_nodes} nodes, {self.graph_data.edge_index.shape[1]} edges")
        return self.graph_data
    
    def load_model(self, model_path: Optional[str] = None):
        """Load or create model"""
        self.model = load_or_create_model(self.config, model_path)
        self.model = self.model.to(self.device)
        return self.model
    
    def run_graph_explanation(self, target_threats: List[str] = None, top_k: int = 5):
        """Run GNNExplainer analysis for graph structure"""
        
        print("\n" + "="*80)
        print(" GRAPH STRUCTURE EXPLANATION (GNNExplainer)")
        print("="*80)
        
        if target_threats is None:
            # Use first few threats as examples
            target_threats = self.column_names[:5]
        
        # Initialize explainer
        graph_explainer = ThreatGraphExplainer(self.model, device=self.device)
        
        explanations = []
        
        for threat_name in target_threats:
            if threat_name in self.column_names:
                node_idx = self.column_names.index(threat_name)
                
                print(f"\nExplaining predictions for: {threat_name} (Node {node_idx})")
                
                # Generate explanation
                explanation = graph_explainer.explain_threat_prediction(
                    self.graph_data, node_idx, threat_name
                )
                
                explanations.append(explanation)
                
                # Visualize explanatory subgraph
                save_path = os.path.join(
                    self.output_dir, 'graphs', f'explanation_{threat_name}.png'
                )
                
                graph_explainer.visualize_explanatory_subgraph(
                    explanation,
                    save_path=save_path
                )
                
                print(f"   Generated explanatory subgraph: {save_path}")
                
                # Print top influential edges
                edges = explanation['influential_edges'][:5]
                print(f"  Top 5 influential edges:")
                for edge in edges:
                    print(f"    {edge['source_name']} → {edge['target_name']} "
                          f"(importance: {edge['importance_score']:.3f}, "
                          f"correlation: {edge['temporal_correlation']:.3f})")
        
        return explanations
    
    def run_attention_analysis(self, target_threats: List[str] = None):
        """Analyze dual-pathway attention in GPSConv layers"""
        
        print("\n" + "="*80)
        print(" DUAL-PATHWAY ATTENTION ANALYSIS")
        print("="*80)
        
        if target_threats is None:
            target_threats = self.column_names[:3]
        
        # Initialize analyzer
        attention_analyzer = GPSConvAttentionAnalyzer(self.model)
        
        for threat_name in target_threats:
            if threat_name in self.column_names:
                node_idx = self.column_names.index(threat_name)
                
                print(f"\nAnalyzing attention for: {threat_name}")
                
                # Run forward pass and collect attention
                attention_result = attention_analyzer.analyze_forward_pass(
                    self.graph_data, node_idx
                )
                
                # Visualize local attention
                if len(attention_result['local_attention']) > 0:
                    save_path = os.path.join(
                        self.output_dir, 'attention', f'local_attention_{threat_name}.png'
                    )
                    
                    attention_analyzer.create_attention_heatmap(
                        attention_result['local_attention'],
                        self.column_names,
                        pathway_type="Local",
                        target_threat=threat_name,
                        save_path=save_path
                    )
                    print(f"   Local attention heatmap: {save_path}")
                
                # Visualize global attention
                if len(attention_result['global_attention']) > 0:
                    save_path = os.path.join(
                        self.output_dir, 'attention', f'global_attention_{threat_name}.png'
                    )
                    
                    attention_analyzer.create_attention_heatmap(
                        attention_result['global_attention'],
                        self.column_names,
                        pathway_type="Global",
                        target_threat=threat_name,
                        save_path=save_path
                    )
                    print(f"   Global attention heatmap: {save_path}")
                
                # Rank influential neighbors
                if len(attention_result['local_attention']) > 0:
                    neighbor_ranking = attention_analyzer.rank_influential_neighbors(
                        attention_result['local_attention'],
                        self.graph_data,
                        node_idx,
                        top_k=10
                    )
                    
                    if not neighbor_ranking.empty:
                        csv_path = os.path.join(
                            self.output_dir, 'attention', f'neighbor_ranking_{threat_name}.csv'
                        )
                        neighbor_ranking.to_csv(csv_path, index=False)
                        print(f"   Neighbor ranking saved: {csv_path}")
                        
                        print("  Top 5 influential neighbors:")
                        print(neighbor_ranking.head().to_string(index=False))
                
                # Compute pathway contributions
                if (len(attention_result['local_attention']) > 0 and 
                    len(attention_result['global_attention']) > 0):
                    
                    contributions = attention_analyzer.compute_pathway_contributions(
                        attention_result['local_attention'],
                        attention_result['global_attention']
                    )
                    
                    print(f"  Local pathway contribution: {contributions['local_contribution_pct']:.1f}%")
                    print(f"  Global pathway contribution: {contributions['global_contribution_pct']:.1f}%")
                    
                    # Save contributions
                    contrib_path = os.path.join(
                        self.output_dir, 'attention', f'pathway_contributions_{threat_name}.json'
                    )
                    with open(contrib_path, 'w') as f:
                        json.dump(contributions, f, indent=2)
        
        return attention_analyzer
    
    def run_attention_rollout(self, target_threats: List[str] = None):
        """Run transformer attention rollout analysis"""
        
        print("\n" + "="*80)
        print(" TRANSFORMER ATTENTION ROLLOUT")
        print("="*80)
        
        if target_threats is None:
            target_threats = self.column_names[:3]
        
        # Initialize rollout analyzer
        rollout_analyzer = TransformerAttentionRollout(self.model, num_layers=self.config['num_layers'])
        
        for threat_name in target_threats:
            if threat_name in self.column_names:
                node_idx = self.column_names.index(threat_name)
                
                print(f"\nRunning attention rollout for: {threat_name}")
                
                # Compute attention rollout
                rollout_attn, layer_attns = rollout_analyzer.compute_attention_rollout(
                    self.graph_data, node_idx
                )
                
                if len(rollout_attn) > 0:
                    # Visualize global attention flow
                    save_path = os.path.join(
                        self.output_dir, 'attention', f'attention_flow_{threat_name}.png'
                    )
                    
                    rollout_analyzer.visualize_global_attention_flow(
                        rollout_attn,
                        self.graph_data,
                        node_idx,
                        threshold=0.05,
                        save_path=save_path
                    )
                    print(f"   Global attention flow: {save_path}")
                    
                    # Create layer-wise attention flow matrix
                    if layer_attns:
                        save_path = os.path.join(
                            self.output_dir, 'attention', f'layer_attention_{threat_name}.png'
                        )
                        
                        rollout_analyzer.create_attention_flow_matrix(
                            layer_attns,
                            self.column_names,
                            save_path=save_path
                        )
                        print(f"   Layer attention heatmaps: {save_path}")
        
        return rollout_analyzer
    
    def run_rwpe_ablation(self):
        """Run RWPE ablation studies"""
        
        print("\n" + "="*80)
        print("RWPE ABLATION STUDIES")
        print("="*80)
        
        # Create model without RWPE (simplified version)
        config_without_pe = self.config.copy()
        config_without_pe['pe_dim'] = 0  # Disable PE
        
        # Adjust node_emb to use full channels since no PE
        class GraphTransformerNoPE(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.pred_len = config['forecast_horizon']
                self.channels = config['channels']
                self.num_nodes = config['num_nodes']
                
                self.node_emb = nn.Linear(config['node_dim'], config['channels'])
                self.edge_emb = nn.Embedding(2, config['channels'])
                
                self.convs = nn.ModuleList()
                for _ in range(config['num_layers']):
                    nn_seq = nn.Sequential(
                        nn.Linear(config['channels'], config['channels']),
                        nn.ReLU(),
                        nn.Linear(config['channels'], config['channels']),
                    )
                    conv = GPSConv(config['channels'], GINEConv(nn_seq), 
                                  heads=config.get('num_heads', 4),
                                  attn_type=config['attn_type'], 
                                  attn_kwargs={'dropout': config['dropout']})
                    self.convs.append(conv)
                
                self.dropout = nn.Dropout(config['dropout'])
                
                self.forecast_head = nn.Sequential(
                    nn.Linear(config['channels'], config['channels'] // 2),
                    nn.ReLU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['channels'] // 2, config['channels'] // 4),
                    nn.ReLU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['channels'] // 4, config['forecast_horizon'] * config['num_nodes']),
                )
            
            def forward(self, x, pe, edge_index, edge_attr, batch=None):
                x = self.node_emb(x)
                edge_attr = self.edge_emb(edge_attr)
                
                if batch is None:
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                
                for conv in self.convs:
                    x = conv(x, edge_index, batch, edge_attr=edge_attr)
                    x = self.dropout(x)
                
                x = global_add_pool(x, batch)
                return self.forecast_head(x)
        
        # Initialize models
        model_with_pe = self.model
        model_without_pe = GraphTransformerNoPE(config_without_pe).to(self.device)
        
        # Initialize analyzer
        rwpe_analyzer = RWPEAblationAnalyzer(model_with_pe, model_without_pe)
        
        # Compute positional importance
        print("\nComputing positional importance...")
        importance_result = rwpe_analyzer.compute_positional_importance(
            self.graph_data, num_iterations=5
        )
        
        # Visualize structural importance
        save_path = os.path.join(
            self.output_dir, 'ablation', 'structural_importance.png'
        )
        
        rwpe_analyzer.create_structural_importance_map(
            self.graph_data,
            importance_result['relative_importance'],
            metric_name="Positional Importance",
            save_path=save_path
        )
        print(f"   Structural importance map: {save_path}")
        
        # Cluster threats by positional importance
        print("\nClustering threats by structural roles...")
        cluster_analysis = rwpe_analyzer.analyze_positional_clusters(
            self.graph_data,
            importance_result['relative_importance'],
            n_clusters=3
        )
        
        cluster_path = os.path.join(
            self.output_dir, 'ablation', 'structural_clusters.csv'
        )
        cluster_analysis.to_csv(cluster_path, index=False)
        print(f"   Structural clusters saved: {cluster_path}")
        
        print("\nTop threats by positional importance:")
        print(cluster_analysis[['Threat_Name', 'Positional_Importance', 'Cluster_Description']].head(10).to_string(index=False))
        
        return rwpe_analyzer, importance_result, cluster_analysis
    
    def run_uncertainty_analysis(self, target_threats: List[str] = None):
        """Run Bayesian uncertainty analysis with MC Dropout"""
        
        print("\n" + "="*80)
        print("BAYESIAN UNCERTAINTY ANALYSIS")
        print("="*80)
        
        if target_threats is None:
            target_threats = self.column_names[:5]
        
        # Initialize uncertainty analyzer
        uncertainty_analyzer = BayesianUncertaintyAnalyzer(self.model, num_samples=50)
        
        # Compute predictive uncertainty for target threats
        target_indices = []
        for threat_name in target_threats:
            if threat_name in self.column_names:
                target_indices.append(self.column_names.index(threat_name))
        
        if target_indices:
            print(f"\nComputing uncertainty for {len(target_indices)} threats...")
            
            uncertainty_result = uncertainty_analyzer.compute_predictive_uncertainty(
                self.graph_data,
                target_nodes=target_indices
            )
            
            # Create confidence matrix
            print("\nCreating confidence matrix for all threats...")
            confidence_matrix = uncertainty_analyzer.compute_uncertainty_confidence_matrix(
                self.graph_data,
                confidence_threshold=0.8
            )
            
            conf_path = os.path.join(
                self.output_dir, 'uncertainty', 'confidence_matrix.csv'
            )
            confidence_matrix.to_csv(conf_path, index=False)
            print(f"   Confidence matrix saved: {conf_path}")
            
            # Visualize confidence heatmap
            save_path = os.path.join(
                self.output_dir, 'uncertainty', 'confidence_heatmap.png'
            )
            
            uncertainty_analyzer.create_confidence_heatmap(
                confidence_matrix,
                save_path=save_path
            )
            print(f"   Confidence heatmap: {save_path}")
            
            # Print top threats by priority
            print("\nTop 10 threats by priority (severity × confidence):")
            top_threats = confidence_matrix.head(10)[['Threat_Name', 'Mean_Prediction', 
                                                       'Confidence_Score', 'Action_Recommendation']]
            print(top_threats.to_string(index=False))
            
            # For each target threat, create timeline visualization
            for i, threat_name in enumerate(target_threats):
                if i < len(target_indices):
                    # Simulate predictions over time (36 months)
                    predictions_over_time = []
                    for month in range(36):
                        pred_dict = {
                            'mean_prediction': np.random.randn(1) * 0.1 + (1 - month/36),
                            'std_prediction': np.random.rand() * 0.2 + 0.1,
                            'lower_bound_95': 0,
                            'upper_bound_95': 0
                        }
                        pred_dict['lower_bound_95'] = pred_dict['mean_prediction'] - 1.96 * pred_dict['std_prediction']
                        pred_dict['upper_bound_95'] = pred_dict['mean_prediction'] + 1.96 * pred_dict['std_prediction']
                        predictions_over_time.append(pred_dict)
                    
                    save_path = os.path.join(
                        self.output_dir, 'uncertainty', f'uncertainty_timeline_{threat_name}.png'
                    )
                    
                    time_labels = [f"M+{m+1}" for m in range(36)]
                    
                    uncertainty_analyzer.visualize_uncertainty_timeline(
                        predictions_over_time,
                        threat_name=threat_name,
                        time_labels=time_labels[:len(predictions_over_time)],
                        save_path=save_path
                    )
                    print(f"   Uncertainty timeline for {threat_name}: {save_path}")
        
        return uncertainty_analyzer, confidence_matrix
    
    def create_xai_report(self):
        """Create comprehensive XAI report"""
        
        print("\n" + "="*80)
        print("GENERATING XAI REPORT")
        print("="*80)
        
        report = {
            'model_config': self.config,
            'data_info': {
                'num_nodes': self.graph_data.num_nodes,
                'num_edges': self.graph_data.edge_index.shape[1],
                'num_threats': len(self.column_names)
            },
            'visualizations': {
                'graph_explanations': [],
                'attention_analysis': [],
                'uncertainty_analysis': [],
                'ablation_studies': []
            }
        }
        
        # List generated files
        for dir_name in ['graphs', 'attention', 'uncertainty', 'ablation']:
            dir_path = os.path.join(self.output_dir, dir_name)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.csv', '.json'))]
                report['visualizations'][dir_name] = files
        
        # Save report
        report_path = os.path.join(self.output_dir, 'xai_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n XAI report saved: {report_path}")
        print(f"\nAll visualizations saved to: {self.output_dir}")
        
        return report
    
    def run_full_pipeline(self, model_path: Optional[str] = None, 
                         target_threats: List[str] = None):
        """Run complete XAI visualization pipeline"""
        
        print("\n" + "="*80)
        print("GRAPH TRANSFORMER XAI VISUALIZATION PIPELINE")
        print("="*80)
        
        # Step 1: Load and prepare data
        print("\n Loading and preparing data...")
        self.load_and_prepare_data()
        
        # Step 2: Load model
        print("\n Loading model...")
        self.load_model(model_path)
        
        # Step 3: Graph explanation
        print("\n Running graph structure explanation...")
        self.run_graph_explanation(target_threats)
        
        # Step 4: Attention analysis
        print("\nRunning attention analysis...")
        self.run_attention_analysis(target_threats)
        
        # Step 5: Attention rollout
        print("\n Running attention rollout...")
        self.run_attention_rollout(target_threats)
        
        # Step 6: Uncertainty analysis
        print("\nRunning uncertainty analysis...")
        self.run_uncertainty_analysis(target_threats)
        
        # Create report
        print("\n Generating XAI report...")
        report = self.create_xai_report()
        
        print("\n" + "="*80)
        print("XAI VISUALIZATION PIPELINE COMPLETE")
        print("="*80)
        
        return report


# Main Execution

def main():
    """Main function to run XAI visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run XAI visualizations for Graph Transformer')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='xai_visualizations_graph',
                       help='Output directory for visualizations')
    parser.add_argument('--model_path', type=str, default='model/GraphTransformer/o_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data_file', type=str, default='sm_data_g.csv',
                       help='Data file name')
    parser.add_argument('--graph_file', type=str, default='graph.csv',
                       help='Graph file name')
    parser.add_argument('--threats', type=str, nargs='+', default=None,
                       help='Specific threats to analyze (space-separated)')
    parser.add_argument('--full_pipeline', action='store_false', default=False,
                       help='Run full XAI pipeline')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = XAIVisualizationPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Load data and graph
    data_dict = load_data_and_graph(
        os.path.join(args.data_dir, args.data_file),
        os.path.join(args.data_dir, args.graph_file)
    )
    
    pipeline.column_names = data_dict['column_names']
    pipeline.config['num_nodes'] = len(pipeline.column_names)
    
    # Create correlation graph
    edge_index, edge_attr, corr_matrix = create_correlation_graph(
        data_dict['data'], pipeline.column_names, threshold=0.3
    )
    pipeline.correlation_matrix = corr_matrix
    
    # Prepare PyG data
    pipeline.graph_data = prepare_pyg_data(
        data_dict['data'], edge_index, edge_attr, pipeline.column_names
    )
    pipeline.graph_data = pipeline.graph_data.to(pipeline.device)
    
    # Load model
    pipeline.load_model(args.model_path)
    
    # Run selected analyses
    #if args.full_pipeline or not (args.threats and len(args.threats) > 0):
        # Run full pipeline
        #pipeline.run_full_pipeline(args.model_path, args.threats)
    #else:
    
    # Run specific analyses for given threats
    print(f"\nRunning XAI analysis for threats: {args.threats}")
    
    #print("\n Graph Structure Explanation...")
    #pipeline.run_graph_explanation(args.threats)
    
    print("\n Attention Analysis...")
    pipeline.run_attention_analysis(args.threats)
    
    print("\n Attention Rollout...")
    pipeline.run_attention_rollout(args.threats)
    
    print("\n Uncertainty Analysis...")
    pipeline.run_uncertainty_analysis(args.threats)
    
    print("\n Generating Report...")
    pipeline.create_xai_report()

    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nTo view specific visualizations:")
    print("  - Graph explanations: xai_visualizations/graphs/")
    print("  - Attention analysis: xai_visualizations/attention/")
    print("  - Uncertainty analysis: xai_visualizations/uncertainty/")
    print("  - Ablation studies: xai_visualizations/ablation/")



if __name__ == "__main__":
    main()