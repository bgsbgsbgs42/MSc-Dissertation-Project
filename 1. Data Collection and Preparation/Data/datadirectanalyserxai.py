"""Script for establishing data-driven XAI analysis to determine data distribution and feature importance without model dependency, e.g. how robust is the data? what outliers are there? which features are most predictive based on data characteristics alone?"""


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

class DirectAnalyzer:
    """
    Direct XAI analysis on data without model dependency
    """
    
    def __init__(self, data_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.results = {}
        
    def load_and_analyze_data(self):
        """Load data and perform direct analysis without model dependency"""
        print("Loading data for direct XAI analysis...")
        
        try:
            # Load data with headers
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            else:
                self.data = pd.read_csv(self.data_path, delim_whitespace=True)
            
            print(f"Data shape: {self.data.shape}")

            # Keep only numeric columns (drop any text-based identifiers)
            numeric_data = self.data.select_dtypes(include=[np.number])
            dropped = set(self.data.columns) - set(numeric_data.columns)
            if dropped:
                print(f"Non-numeric columns dropped: {', '.join(dropped)}")

            self.data = numeric_data

            # Use actual column headers as feature names
            self.feature_names = list(self.data.columns)

            # Normalize data
            self.scaler = StandardScaler()
            self.data_normalized = self.scaler.fit_transform(self.data)

            print("Data successfully loaded and normalized with real column headers.")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def correlation_analysis(self):
        """Perform correlation-based feature importance"""
        print("Performing correlation analysis...")
        
        os.makedirs('results/correlation', exist_ok=True)
        
        try:
            # Calculate correlations between features
            corr_matrix = np.corrcoef(self.data_normalized.T)
            
            # Feature importance based on correlation with future values
            # (using time-shifted correlations as proxy for predictive power)
            importance_scores = []
            
            for i in range(self.data.shape[1]):
                # Correlation with future values (simple time series importance)
                if len(self.data) > 1:
                    # Use correlation with next time step as importance measure
                    corr_with_future = np.corrcoef(self.data_normalized[:-1, i], 
                                                  self.data_normalized[1:, i])[0, 1]
                    importance_scores.append(abs(corr_with_future))
                else:
                    # Fallback: variance as importance
                    importance_scores.append(np.var(self.data_normalized[:, i]))
            
            # Get top features
            top_k = min(15, len(importance_scores))
            top_indices = np.argsort(importance_scores)[-top_k:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_scores = [importance_scores[i] for i in top_indices]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_scores)), top_scores)
            plt.yticks(range(len(top_scores)), top_features)
            plt.xlabel('Importance Score (Absolute Correlation)')
            plt.title('Feature Importance based on Temporal Correlation')
            plt.tight_layout()
            plt.savefig('results/correlation/feature_importance.png', dpi=300)
            plt.close()
            
            # Plot correlation matrix heatmap for top features
            plt.figure(figsize=(10, 8))
            top_corr_matrix = corr_matrix[top_indices][:, top_indices]
            sns.heatmap(top_corr_matrix, 
                       xticklabels=top_features,
                       yticklabels=top_features,
                       cmap='coolwarm', center=0,
                       annot=True, fmt='.2f')
            plt.title('Correlation Matrix of Top Features')
            plt.tight_layout()
            plt.savefig('results/correlation/correlation_heatmap.png', dpi=300)
            plt.close()
            
            result = {
                'top_features': top_features,
                'importance_scores': top_scores,
                'correlation_matrix': corr_matrix.tolist()
            }
            
            self.results['correlation'] = result
            print("Correlation analysis completed")
            return result
            
        except Exception as e:
            print(f"Correlation analysis failed: {e}")
            return None
    
    def statistical_analysis(self):
        """Perform statistical analysis of the data"""
        print("Performing statistical analysis...")
        
        os.makedirs('results/statistical', exist_ok=True)
        
        try:
            # Calculate various statistical measures
            statistics = {}
            
            # Basic statistics
            statistics['mean'] = np.mean(self.data_normalized, axis=0).tolist()
            statistics['std'] = np.std(self.data_normalized, axis=0).tolist()
            statistics['variance'] = np.var(self.data_normalized, axis=0).tolist()
            
            # Temporal statistics (change over time)
            if len(self.data_normalized) > 1:
                differences = np.diff(self.data_normalized, axis=0)
                statistics['avg_change'] = np.mean(np.abs(differences), axis=0).tolist()
                statistics['volatility'] = np.std(differences, axis=0).tolist()
            
            # Identify outliers using IQR method
            Q1 = np.percentile(self.data_normalized, 25, axis=0)
            Q3 = np.percentile(self.data_normalized, 75, axis=0)
            IQR = Q3 - Q1
            outlier_mask = (self.data_normalized < (Q1 - 1.5 * IQR)) | (self.data_normalized > (Q3 + 1.5 * IQR))
            statistics['outlier_percentage'] = (np.sum(outlier_mask, axis=0) / len(self.data_normalized) * 100).tolist()
            
            # Plot statistical distributions
            num_features_to_plot = min(6, self.data.shape[1])
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i in range(num_features_to_plot):
                axes[i].hist(self.data_normalized[:, i], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{self.feature_names[i]} Distribution')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
            
            for i in range(num_features_to_plot, 6):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig('results/statistical/feature_distributions.png', dpi=300)
            plt.close()
            
            # Plot outlier analysis
            plt.figure(figsize=(12, 6))
            features_to_show = min(10, len(statistics['outlier_percentage']))
            indices = np.argsort(statistics['outlier_percentage'])[-features_to_show:][::-1]
            outlier_percentages = [statistics['outlier_percentage'][i] for i in indices]
            feature_names = [self.feature_names[i] for i in indices]
            
            plt.bar(range(len(outlier_percentages)), outlier_percentages)
            plt.xticks(range(len(outlier_percentages)), feature_names, rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Outlier Percentage (%)')
            plt.title('Percentage of Outliers by Feature')
            plt.tight_layout()
            plt.savefig('results/statistical/outlier_analysis.png', dpi=300)
            plt.close()
            
            self.results['statistical'] = statistics
            print("Statistical analysis completed")
            return statistics
            
        except Exception as e:
            print(f"Statistical analysis failed: {e}")
            return None
    
    def time_series_analysis(self):
        """Analyze time series patterns in the data"""
        print("Performing time series analysis...")
        
        os.makedirs('results/time_series', exist_ok=True)
        
        try:
            # Analyze temporal patterns
            time_series_stats = {}
            
            # Autocorrelation analysis
            autocorr_lags = min(10, len(self.data_normalized) // 4)
            autocorrelations = []
            
            for feature_idx in range(min(10, self.data.shape[1])):
                feature_data = self.data_normalized[:, feature_idx]
                autocorr = []
                for lag in range(1, autocorr_lags + 1):
                    if len(feature_data) > lag:
                        corr = np.corrcoef(feature_data[:-lag], feature_data[lag:])[0, 1]
                        autocorr.append(corr)
                    else:
                        autocorr.append(0)
                autocorrelations.append(autocorr)
            
            time_series_stats['autocorrelations'] = autocorrelations
            
            # Plot autocorrelations for top features
            num_features_to_plot = min(5, len(autocorrelations))
            plt.figure(figsize=(12, 8))
            
            for i in range(num_features_to_plot):
                plt.plot(range(1, autocorr_lags + 1), autocorrelations[i], 
                        marker='o', label=f'Feature {i}')
            
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Autocorrelation of Top Features')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/time_series/autocorrelation.png', dpi=300)
            plt.close()
            
            # Plot time series of most important features (based on variance)
            variances = np.var(self.data_normalized, axis=0)
            top_var_indices = np.argsort(variances)[-6:][::-1]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature_idx in enumerate(top_var_indices):
                axes[i].plot(self.data_normalized[:, feature_idx])
                axes[i].set_title(f'{self.feature_names[feature_idx]} (var: {variances[feature_idx]:.3f})')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/time_series/feature_trajectories.png', dpi=300)
            plt.close()
            
            # Trend analysis
            if len(self.data_normalized) > 5:
                trends = []
                for feature_idx in range(min(10, self.data.shape[1])):
                    x = np.arange(len(self.data_normalized))
                    y = self.data_normalized[:, feature_idx]
                    z = np.polyfit(x, y, 1)
                    trends.append(z[0])  # Slope
                
                time_series_stats['trends'] = trends
                
                # Plot trends
                plt.figure(figsize=(10, 6))
                feature_indices = range(min(10, len(trends)))
                plt.bar(feature_indices, [trends[i] for i in feature_indices])
                plt.xlabel('Feature Index')
                plt.ylabel('Trend Slope')
                plt.title('Linear Trend of Features Over Time')
                plt.xticks(feature_indices, [f'F{i}' for i in feature_indices])
                plt.tight_layout()
                plt.savefig('results/time_series/trend_analysis.png', dpi=300)
                plt.close()
            
            self.results['time_series'] = time_series_stats
            print(" Time series analysis completed")
            return time_series_stats
            
        except Exception as e:
            print(f" Time series analysis failed: {e}")
            return None
    
    def clustering_analysis(self):
        """Perform clustering to identify patterns in the data"""
        print("Performing clustering analysis...")
        
        os.makedirs('results/clustering', exist_ok=True)
        
        try:
            
            
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(self.data_normalized)
            
            # Perform clustering
            n_clusters = min(5, len(self.data_normalized) // 10)
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.data_normalized)
            
            # Plot clusters in 2D space
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Data Clustering in 2D PCA Space')
            plt.tight_layout()
            plt.savefig('results/clustering/pca_clusters.png', dpi=300)
            plt.close()
            
            # Analyze cluster characteristics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = self.data_normalized[clusters == cluster_id]
                if len(cluster_data) > 0:
                    cluster_stats[cluster_id] = {
                        'size': len(cluster_data),
                        'mean_features': np.mean(cluster_data, axis=0).tolist(),
                        'std_features': np.std(cluster_data, axis=0).tolist()
                    }
            
            # Plot cluster sizes
            plt.figure(figsize=(8, 6))
            cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
            plt.bar(range(len(cluster_sizes)), cluster_sizes)
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Points')
            plt.title('Cluster Sizes')
            plt.xticks(range(len(cluster_sizes)))
            plt.tight_layout()
            plt.savefig('results/clustering/cluster_sizes.png', dpi=300)
            plt.close()
            
            # Feature importance based on cluster separation
            feature_importance = []
            for feature_idx in range(self.data.shape[1]):
                # Calculate between-cluster variance / within-cluster variance
                overall_mean = np.mean(self.data_normalized[:, feature_idx])
                between_var = 0
                within_var = 0
                
                for cluster_id, stats in cluster_stats.items():
                    cluster_mean = stats['mean_features'][feature_idx]
                    cluster_size = stats['size']
                    between_var += cluster_size * (cluster_mean - overall_mean) ** 2
                    
                    cluster_data = self.data_normalized[clusters == cluster_id, feature_idx]
                    within_var += np.sum((cluster_data - cluster_mean) ** 2)
                
                if within_var > 0:
                    importance = between_var / within_var
                else:
                    importance = 0
                
                feature_importance.append(importance)
            
            # Plot clustering-based feature importance
            top_k = min(15, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_scores = [feature_importance[i] for i in top_indices]
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(top_scores)), top_scores)
            plt.yticks(range(len(top_scores)), top_features)
            plt.xlabel('Cluster Separation Importance')
            plt.title('Feature Importance based on Cluster Separation')
            plt.tight_layout()
            plt.savefig('results/clustering/clustering_importance.png', dpi=300)
            plt.close()
            
            result = {
                'clusters': cluster_stats,
                'feature_importance': feature_importance,
                'pca_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
            
            self.results['clustering'] = result
            print("Clustering analysis completed")
            return result
            
        except Exception as e:
            print(f"Clustering analysis failed: {e}")
            return None
    
    def generate_comprehensive_report(self):
        """Generate comprehensive XAI report based on data analysis"""
        print("Generating comprehensive report...")
        
        os.makedirs('results', exist_ok=True)
        
        # Create summary report
        report = {
            'data_info': {
                'shape': self.data.shape,
                'num_features': self.data.shape[1],
                'num_samples': self.data.shape[0]
            },
            'analysis_results': {}
        }
        
        # Summarize each analysis
        for method, result in self.results.items():
            if method == 'correlation' and 'top_features' in result:
                report['analysis_results'][method] = {
                    'top_5_features': result['top_features'][:5],
                    'status': 'completed'
                }
            elif method == 'statistical':
                report['analysis_results'][method] = {
                    'features_analyzed': len(result.get('mean', [])),
                    'status': 'completed'
                }
            elif method == 'time_series':
                report['analysis_results'][method] = {
                    'autocorrelation_lags': len(result.get('autocorrelations', [[]])[0]) if result.get('autocorrelations') else 0,
                    'status': 'completed'
                }
            elif method == 'clustering':
                report['analysis_results'][method] = {
                    'num_clusters': len(result.get('clusters', {})),
                    'status': 'completed'
                }
        
        # Save JSON report
        with open('results/comprehensive_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create executive summary
        with open('results/executive_summary.txt', 'w') as f:
            f.write("DATA-DRIVEN XAI ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Shape: {self.data.shape[0]} samples × {self.data.shape[1]} features\n")
            f.write(f"Analyses completed: {len(self.results)}\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            
            if 'correlation' in self.results:
                top_features = self.results['correlation'].get('top_features', [])[:3]
                f.write(f"• Top predictive features: {', '.join(top_features)}\n")
            
            if 'statistical' in self.results:
                outlier_pct = self.results['statistical'].get('outlier_percentage', [])
                if outlier_pct:
                    max_outliers = max(outlier_pct)
                    f.write(f"• Maximum outlier percentage: {max_outliers:.1f}%\n")
            
            if 'clustering' in self.results:
                num_clusters = len(self.results['clustering'].get('clusters', {}))
                f.write(f"• Natural data clusters identified: {num_clusters}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            f.write("1. Focus on top correlated features for threat prediction\n")
            f.write("2. Monitor high-volatility features for anomaly detection\n")
            f.write("3. Use clustering patterns to identify threat categories\n")
            f.write("4. Consider temporal patterns for proactive threat detection\n")
        
        print(" Comprehensive report generated")
    
    def run_complete_analysis(self):
        """Run complete data-driven XAI analysis"""
        print("Starting data-driven XAI analysis...")
        print("=" * 60)
        
        # Load data
        if not self.load_and_analyze_data():
            print(" Failed to load data")
            return False
        
        print("\n" + "=" * 60)
        print("RUNNING DATA ANALYSIS METHODS")
        print("=" * 60)
        
        # Run analyses
        methods = [
            ("Correlation Analysis", self.correlation_analysis),
            ("Statistical Analysis", self.statistical_analysis),
            ("Time Series Analysis", self.time_series_analysis),
            ("Clustering Analysis", self.clustering_analysis),
        ]
        
        successful_methods = 0
        for method_name, method_func in methods:
            print(f"\n--- {method_name} ---")
            try:
                result = method_func()
                if result is not None:
                    print(f" SUCCESS: {method_name}")
                    successful_methods += 1
                else:
                    print(f"  PARTIAL: {method_name}")
            except Exception as e:
                print(f" FAILED: {method_name} - {e}")
        
        # Generate report
        print(f"\n--- Generating Report ---")
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Successful analyses: {successful_methods}/{len(methods)}")
        print(f"Results saved in: results/")
        
        return successful_methods > 0

def main():
    """Main function"""
    
    # Update this path to your data file
    data_path = './data/sm_data_g.csv'
    
    # Try alternative paths
    alternative_paths = [
        './data/sm_data_g.csv',
        '../data/sm_data_g.csv',
        './sm_data_g.csv',
        'sm_data_g.csv'
    ]
    
    working_path = None
    for path in alternative_paths:
        if os.path.exists(path):
            working_path = path
            print(f"Found data file: {path}")
            break
    
    if working_path is None:
        print("Could not find data file. Please check the path.")
        return
    
    # Run analysis
    analyzer = DirectAnalyzer(working_path)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n Data-driven XAI analysis completed successfully!")
        print("Check the 'results/' directory for all outputs")
    else:
        print("\n Analysis completed with some limitations.")
        print("Basic results are available in 'results/'")

if __name__ == "__main__":
    main()