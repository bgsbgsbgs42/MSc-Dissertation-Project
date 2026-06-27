import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import shap
import lime
import lime.lime_tabular
import dice_ml as dice
import networkx as nx
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import openai
from langchain_core.messages import HumanMessage, SystemMessage
#from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pgmpy.estimators import PC, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import scipy.stats as stats
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict
from ModelSelectionAgent import ModelSelectionAgent
from AnalysisAgent import AnalysisAgent
from LSTMForecaster import LSTMForecaster
from RAGSystem import RAGSystem
from XAIAnalyzer import XAIAnalyzer
from VisualizationAgent import VisualizationAgent
from PATRecommendationAgent import PATRecommendationAgent, PATConfig
from preprocess_agent import PreprocessAgent
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastConfig:
    """Configuration for the forecasting system"""
    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "deepseek-r1:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    
    # Data Configuration
    data_path: str = "data/sm_data_g.csv"
    graph_path: str = "data/graph.csv"
    forecast_months: int = 36
    target_columns: List[str] = None
    
    # Model Configuration
    validation_split: float = 0.8
    seasonal_period: int = 12
    
    # Output Configuration
    output_dir: str = "forecast_results"
    plot_style: str = "default"
    
    # Agent Configuration
    enable_agents: bool = True
    max_retries: int = 3
    
    # XAI Configuration
    enable_xai: bool = True
    xai_methods: List[str] = None
    
    # RAG Configuration
    enable_rag: bool = False
    rag_db_path: str = "cyber_threat_db"

    # Report Configuration
    enab_report: bool = True
    report_path: str = "forecast_report.pdf"    
    
    # New Models
    enable_lstm: bool = True
    enable_prophet: bool = True
    enable_validation: bool = True  
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
            'n_candidates': 3,
            'k_models': 3, 
            'cv_folds': 5,
            'optimization_method': 'grid_search'
    })
    
    # Preprocessing Configuration
    enable_preprocessing: bool = True
    preprocess_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize target columns after object creation"""
        if self.target_columns is None:
            # This will be populated after loading graph.csv
            self.target_columns = []
            
        # Initialize preprocessing configuration
        if self.preprocess_config is None:
            self.preprocess_config = {
                'preprocess': {
                    'outlier_threshold': 1.5,
                    'enable_llm_preprocessing': True
                }
            }
            
        # Initialize XAI methods with all available methods
        if self.xai_methods is None:
            self.xai_methods = [
                "shap",           # SHAP analysis
                "lime",           # LIME analysis  
                "attention",      # Attention visualization
                "permutation",    # Permutation feature importance
                "counterfactual", # Counterfactual analysis
                "causal",         # Causal network analysis
                "faithfulness",   # Explanation faithfulness
                "dynamic_weights", # Dynamic weight interpretation
                "dice",           # DiCE counterfactuals
                "consensus",      # Multi-model consensus
                "anchors"         # Rule-based explanations
            ]

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super().default(obj)
class LLMManager:
    """LLM-agnostic manager for different model providers"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.config.llm_provider == "ollama":
            return Ollama(
                model=self.config.llm_model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
            
        """elif self.config.llm_provider == "openai":
            if self.config.api_key:
                openai.api_key = self.config.api_key
            return ChatOpenAI(
                model_name=self.config.llm_model,
                temperature=0.1,
                max_tokens=4000
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=0.1,
                max_tokens=4000
            )"""
        
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the configured LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

class ReportAgent:
    """
    Self-contained Report Agent for generating comprehensive final reports.
    Responsibilities:
    - Synthesize all experiment results
    - Create executive summaries  
    - Generate actionable recommendations
    - Produce final visualizations
    - Save results in multiple formats
    """
    
    def __init__(self, llm_manager=None, config: Dict[str, Any] = None):
        self.llm_manager = llm_manager
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'reports'))
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report with all findings and recommendations"""
        logger.info("Generating comprehensive final report...")
        
        report_data = {
            "executive_summary": self._generate_executive_summary(results),
            "technical_analysis": self._generate_technical_analysis(results),
            "model_performance": self._analyze_model_performance(results),
            "cybersecurity_insights": self._generate_cybersecurity_insights(results),
            "recommendations": self._generate_recommendations(results),
            "visualizations": self._generate_report_visualizations(results),
            "metadata": self._generate_report_metadata(results)
        }
        
        # Generate LLM-enhanced report if LLM is available
        if self.llm_manager:
            report_data["llm_enhanced_report"] = self._generate_llm_enhanced_report(results)
        
        # Save all report components
        self._save_report_components(report_data, results)
        
        logger.info(f"Comprehensive report generated and saved to: {self.output_dir}")
        return report_data
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the configured LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""


    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for stakeholders"""
        forecasts = results.get("forecasts", {})
        performance = results.get("training_results", {}).get("performance", {})
        pat_recommendations = results.get("pat_recommendations", {})
        
        # Calculate overall metrics
        total_nodes = len(forecasts)
        models_used = set()
        avg_performance = {}
        
        for node_metrics in performance.values():
            for model_name, metrics in node_metrics.items():
                models_used.add(model_name)
                for metric, value in metrics.items():
                    if metric not in avg_performance:
                        avg_performance[metric] = []
                    avg_performance[metric].append(value)
        
        # Calculate averages robustly
        for metric, values in avg_performance.items():
            cleaned_values = []

            for v in values:
                if isinstance(v, dict):
                    numeric_parts = [
                        x for x in v.values()
                        if isinstance(x, (int, float, np.number))
                    ]
                    cleaned_values.extend(numeric_parts)
                elif isinstance(v, (int, float, np.number)):
                    cleaned_values.append(v)

            if cleaned_values:
                avg_performance[metric] = float(np.mean(cleaned_values))
            else:
                avg_performance[metric] = None
        
        # PAT recommendations summary
        pat_summary = {}
        if pat_recommendations and pat_recommendations.get('recommended_pats'):
            best_combo = pat_recommendations['recommended_pats'][0]
            pat_summary = {
                "recommended_pats_count": len(best_combo.get('pats', [])),
                "estimated_cost": best_combo.get('total_cost', 0),
                "risk_reduction": pat_recommendations.get('risk_reduction_metrics', {}).get('risk_reduction_percentage', 0)
            }
        
        return {
            "total_nodes_analyzed": total_nodes,
            "models_used": list(models_used),
            "key_metrics": {
                k: v for k, v in avg_performance.items()
                if k in ["MAE", "RMSE", "RRSE", "MAPE", "RAE", "Operational_Readiness_Index"]
            },
            "pat_recommendations_summary": pat_summary,
            "overall_assessment": self._assemble_overall_assessment(results),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_technical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical analysis"""
        analysis_results = results.get("analysis_results", {})
        training_results = results.get("training_results", {})
        xai_results = results.get("xai_results", {})
        
        technical_analysis = {
            "data_characteristics": analysis_results.get("data_characteristics", {}),
            "model_performance_comparison": self._compare_model_performance(training_results),
            "xai_insights": self._extract_xai_insights(xai_results),
            "validation_results": results.get("validation_results", {}),
            "preprocessing_impact": results.get("preprocessing_results", {})
        }
        
        return technical_analysis
    
    def _analyze_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare model performance across all nodes"""
        performance = results.get("training_results", {}).get("performance", {})
        model_comparison: Dict[str, Dict[str, list]] = {}
    
        # Collect raw metric values per model
        for node, node_metrics in performance.items():
            for model_name, metrics in node_metrics.items():
                if model_name not in model_comparison:
                    model_comparison[model_name] = {}
                
            for metric, value in metrics.items():
                if metric not in model_comparison[model_name]:
                    model_comparison[model_name][metric] = []
                
                #  normalize value into numeric pieces 
                if isinstance(value, dict):
                    # e.g. per-horizon dict: {"h1": 0.1, "h2": 0.2, ...}
                    for v in value.values():
                        if isinstance(v, (int, float, np.number)):
                            model_comparison[model_name][metric].append(v)
                elif isinstance(value, (int, float, np.number)):
                    model_comparison[model_name][metric].append(value)
                # else: ignore non-numeric (strings, None, etc.)
    
        # Calculate statistics for each model
        model_stats: Dict[str, Dict[str, float]] = {}
        for model_name, metrics in model_comparison.items():
            model_stats[model_name] = {}
            for metric, values in metrics.items():
                # Filter to numeric values only in case anything weird slipped through
                cleaned_values = [
                    v for v in values if isinstance(v, (int, float, np.number))
                ]
                if not cleaned_values:
                    continue  # nothing usable for this metric
            
                arr = np.asarray(cleaned_values, dtype=float)
                model_stats[model_name][f"{metric}_mean"] = float(np.mean(arr))
                model_stats[model_name][f"{metric}_std"]  = float(np.std(arr))
                model_stats[model_name][f"{metric}_min"]  = float(np.min(arr))
                model_stats[model_name][f"{metric}_max"]  = float(np.max(arr))
    
        # Rank models by key metrics 
        ranked_models = self._rank_models(model_stats)
    
        return {
            "model_statistics": model_stats,
            "ranked_models": ranked_models,
            "best_performing_models": ranked_models[:3] if ranked_models else []
        }

    def _generate_cybersecurity_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cybersecurity-specific insights"""
        forecasts = results.get("forecasts", {})
        performance = results.get("training_results", {}).get("performance", {})
        pat_recommendations = results.get("pat_recommendations", {})
        
        # Analyze threat patterns
        threat_analysis = self._analyze_threat_patterns(forecasts, performance)
        
        # Risk assessment
        risk_assessment = self._assess_cybersecurity_risk(forecasts, pat_recommendations)
        
        # Operational readiness
        operational_readiness = self._assess_operational_readiness(performance)
        
        return {
            "threat_analysis": threat_analysis,
            "risk_assessment": risk_assessment,
            "operational_readiness": operational_readiness,
            "key_vulnerabilities": self._identify_key_vulnerabilities(forecasts)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        model_performance = self._analyze_model_performance(results)
        cybersecurity_insights = self._generate_cybersecurity_insights(results)
        pat_recommendations = results.get("pat_recommendations", {})
        
        recommendations = {
            "model_selection": self._generate_model_recommendations(model_performance),
            "cybersecurity_actions": self._generate_cybersecurity_actions(cybersecurity_insights),
            "pat_implementation": self._generate_pat_implementation_plan(pat_recommendations),
            "monitoring_suggestions": self._generate_monitoring_suggestions(results),
            "future_improvements": self._generate_future_improvements(results)
        }
        
        return recommendations
    
    def _generate_report_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for the report"""
        visualizations = {}
        
        try:
            # Performance comparison chart
            perf_chart_path = self.output_dir / "model_performance_comparison.png"
            self._create_performance_chart(results, str(perf_chart_path))
            visualizations["performance_comparison"] = str(perf_chart_path)
            
            # Threat forecast visualization
            threat_chart_path = self.output_dir / "threat_forecast_overview.png"
            self._create_threat_forecast_chart(results, str(threat_chart_path))
            visualizations["threat_forecast"] = str(threat_chart_path)
            
            # PAT recommendations chart
            if results.get("pat_recommendations"):
                pat_chart_path = self.output_dir / "pat_recommendations_summary.png"
                self._create_pat_recommendations_chart(results, str(pat_chart_path))
                visualizations["pat_recommendations"] = str(pat_chart_path)
                
        except Exception as e:
            logger.error(f"Error generating report visualizations: {e}")
            
        return visualizations
    
    def _generate_llm_enhanced_report(self, results: Dict[str, Any]) -> str:
        """Generate LLM-enhanced comprehensive report"""
        if not self.llm_manager:
            return "LLM-enhanced report not available (no LLM manager provided)"
            
        system_prompt = """You are an expert cybersecurity forecasting analyst. Generate a comprehensive, insightful report that synthesises all forecasting results, model performance, cybersecurity insights, and recommendations. Focus on actionable insights for cybersecurity stakeholders."""
        
        user_prompt = f"""
        Based on the following cybersecurity forecasting results, generate a comprehensive report:
        
        {json.dumps(self._prepare_results_for_llm(results), indent=2)}
        
        Please structure your report with:
        1. Executive Summary
        2. Key Findings and Insights
        3. Model Performance Analysis
        4. Cybersecurity Threat Assessment
        5. PAT Recommendations Summary
        6. Actionable Recommendations
        7. Risk Assessment and Mitigation Strategies
        
        Make the report professional, data-driven, and focused on cybersecurity operational impact.
        """
        
        try:
            llm_report = self.llm_manager.generate_response(system_prompt, user_prompt)
            return llm_report
        except Exception as e:
            logger.error(f"Error generating LLM-enhanced report: {e}")
            return "LLM-enhanced report generation failed"
    
    # Helper methods for report generation
    def _assemble_overall_assessment(self, results: Dict[str, Any]) -> str:
        """Assemble overall assessment based on results"""
        performance = results.get("training_results", {}).get("performance", {})
        
        if not performance:
            return "Insufficient data for assessment"
            
        # Calculate average operational readiness
        ori_scores = []
        for node_metrics in performance.values():
            for metrics in node_metrics.values():
                if "Operational_Readiness_Index" in metrics:
                    ori_scores.append(metrics["Operational_Readiness_Index"])
        
        avg_ori = np.mean(ori_scores) if ori_scores else 0
        
        if avg_ori > 0.8:
            return "EXCELLENT - High operational readiness with strong predictive capabilities"
        elif avg_ori > 0.6:
            return "GOOD - Solid performance with reliable threat forecasting"
        elif avg_ori > 0.4:
            return "FAIR - Moderate performance, some areas need improvement"
        else:
            return "POOR - Significant improvements needed in forecasting accuracy"
    
    def _compare_model_performance(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different models"""
        performance = training_results.get("performance", {})
        model_scores = {}
        
        for node, node_metrics in performance.items():
            for model_name, metrics in node_metrics.items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                
                # Use operational readiness as primary score
                score = metrics.get("Operational_Readiness_Index", 
                                 0.5 * (1 - metrics.get("RMSE", 1)) + 0.3 * (1 - metrics.get("MAE", 1)) + 0.2 * metrics.get("Attack_Coverage", 0))
                model_scores[model_name].append(score)
        
        # Calculate average scores
        avg_scores = {model: np.mean(scores) for model, scores in model_scores.items()}
        
        # Sort by performance
        ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "average_scores": avg_scores,
            "ranking": ranked_models,
            "best_model": ranked_models[0] if ranked_models else None
        }
    
    def _extract_xai_insights(self, xai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from XAI analysis"""
        insights = {}
        
        if not xai_results:
            return {"status": "No XAI results available"}
        
        # Extract feature importance insights
        if "feature_importance" in xai_results:
            insights["key_features"] = xai_results["feature_importance"].get("top_features", [])
        
        # Extract model consensus insights
        if "multi_model_consensus" in xai_results:
            consensus = xai_results["multi_model_consensus"]
            insights["high_agreement_features"] = consensus.get("high_agreement_features", [])
            insights["model_disagreements"] = consensus.get("disagreement_features", [])
        
        # Extract counterfactual insights
        if "counterfactual_analysis" in xai_results:
            counterfactuals = xai_results["counterfactual_analysis"]
            insights["critical_thresholds"] = counterfactuals.get("critical_thresholds", {})
        
        return insights
    
    def _rank_models(self, model_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models based on multiple performance metrics"""
        ranked_models = []
        
        for model_name, stats in model_stats.items():
            # Calculate composite score (weighted combination of key metrics)
            ori_score = stats.get("Operational_Readiness_Index_mean", 0.5)
            rmse_score = 1 - min(stats.get("RMSE_mean", 1) / 10, 1)  # Normalize RMSE
            coverage_score = stats.get("Attack_Coverage_mean", 0.5)
            fpr_score = 1 - stats.get("False_Positive_Rate_mean", 0.5)
            
            composite_score = (0.4 * ori_score + 0.3 * rmse_score + 
                             0.2 * coverage_score + 0.1 * fpr_score)
            
            ranked_models.append({
                "model": model_name,
                "composite_score": composite_score,
                "operational_readiness": ori_score,
                "rmse": stats.get("RMSE_mean", None),
                "attack_coverage": coverage_score,
                "false_positive_rate": stats.get("False_Positive_Rate_mean", None)
            })
        
        # Sort by composite score
        return sorted(ranked_models, key=lambda x: x["composite_score"], reverse=True)
    
    def _analyze_threat_patterns(self, forecasts: Dict[str, Any], performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cybersecurity threat patterns from forecasts"""
        threat_analysis = {
            "high_risk_nodes": [],
            "emerging_threats": [],
            "seasonal_patterns": [],
            "correlation_insights": []
        }
        
        # Identify high-risk nodes based on forecast values
        for node, model_forecasts in forecasts.items():
            if model_forecasts:
                # Use the first available model's forecast
                for forecast_values in model_forecasts.values():
                    if isinstance(forecast_values, list) and len(forecast_values) > 0:
                        avg_forecast = np.mean(forecast_values)
                        max_forecast = np.max(forecast_values)
                        
                        if max_forecast > np.percentile([np.max(v) for v in model_forecasts.values() if v], 75):
                            threat_analysis["high_risk_nodes"].append({
                                "node": node,
                                "average_forecast": avg_forecast,
                                "peak_forecast": max_forecast,
                                "risk_level": "HIGH" if max_forecast > 0.7 else "MEDIUM"
                            })
                        break
        
        return threat_analysis
    
    def _assess_cybersecurity_risk(self, forecasts: Dict[str, Any], pat_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall cybersecurity risk"""
        risk_factors = []
        total_risk_score = 0
        
        # Calculate risk from forecasts
        for node, model_forecasts in forecasts.items():
            if model_forecasts:
                for forecast_values in model_forecasts.values():
                    if isinstance(forecast_values, list):
                        node_risk = np.mean(forecast_values)
                        risk_factors.append({
                            "node": node,
                            "risk_score": node_risk,
                            "contribution": node_risk
                        })
                        total_risk_score += node_risk
                        break
        
        # Adjust risk based on PAT recommendations
        risk_mitigation = 0
        if pat_recommendations and pat_recommendations.get('risk_reduction_metrics'):
            risk_mitigation = pat_recommendations['risk_reduction_metrics'].get('risk_reduction_percentage', 0) / 100
        
        residual_risk = total_risk_score * (1 - risk_mitigation)
        
        return {
            "total_risk_score": total_risk_score,
            "residual_risk_score": residual_risk,
            "risk_mitigation_percentage": risk_mitigation * 100,
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if residual_risk > 0.7 else "MEDIUM" if residual_risk > 0.4 else "LOW"
        }
    
    def _assess_operational_readiness(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational readiness based on model performance"""
        ori_scores = []
        coverage_scores = []
        fpr_scores = []
        
        for node_metrics in performance.values():
            for metrics in node_metrics.values():
                ori_scores.append(metrics.get("Operational_Readiness_Index", 0))
                coverage_scores.append(metrics.get("Attack_Coverage", 0))
                fpr_scores.append(metrics.get("False_Positive_Rate", 0))
        
        avg_ori = np.mean(ori_scores) if ori_scores else 0
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0
        avg_fpr = np.mean(fpr_scores) if fpr_scores else 0
        
        return {
            "average_operational_readiness": avg_ori,
            "average_attack_coverage": avg_coverage,
            "average_false_positive_rate": avg_fpr,
            "readiness_level": "HIGH" if avg_ori > 0.7 else "MEDIUM" if avg_ori > 0.5 else "LOW"
        }
    
    def _identify_key_vulnerabilities(self, forecasts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key vulnerabilities from forecast data"""
        vulnerabilities = []
        
        for node, model_forecasts in forecasts.items():
            if model_forecasts:
                for forecast_values in model_forecasts.values():
                    if isinstance(forecast_values, list) and len(forecast_values) > 0:
                        # Simple vulnerability scoring based on forecast trends
                        trend = np.polyfit(range(len(forecast_values)), forecast_values, 1)[0]
                        volatility = np.std(forecast_values)
                        
                        if trend > 0.1 or volatility > 0.3:  # Adjust thresholds as needed
                            vulnerabilities.append({
                                "node": node,
                                "trend_direction": "INCREASING" if trend > 0 else "DECREASING",
                                "trend_magnitude": abs(trend),
                                "volatility": volatility,
                                "risk_level": "HIGH" if trend > 0.2 else "MEDIUM"
                            })
                        break
        
        return sorted(vulnerabilities, key=lambda x: x["trend_magnitude"], reverse=True)[:10]  # Top 10
    
    def _generate_model_recommendations(self, model_performance: Dict[str, Any]) -> List[str]:
        """Generate model selection recommendations"""
        recommendations = []
        best_models = model_performance.get("best_performing_models", [])
        
        if best_models:
            top_model = best_models[0]
            recommendations.append(f"Primary model: {top_model['model']} (composite score: {top_model['composite_score']:.3f})")
            
            if len(best_models) > 1:
                recommendations.append(f"Backup model: {best_models[1]['model']} for diversity and robustness")
        
        # General recommendations
        recommendations.extend([
            "Implement ensemble approaches for critical nodes",
            "Regularly revalidate model performance with new data",
            "Consider model interpretability for compliance requirements"
        ])
        
        return recommendations
    
    def _generate_cybersecurity_actions(self, cybersecurity_insights: Dict[str, Any]) -> List[str]:
        """Generate cybersecurity action recommendations"""
        actions = []
        threat_analysis = cybersecurity_insights.get("threat_analysis", {})
        risk_assessment = cybersecurity_insights.get("risk_assessment", {})
        
        # Actions based on high-risk nodes
        high_risk_nodes = threat_analysis.get("high_risk_nodes", [])
        if high_risk_nodes:
            actions.append(f"Prioritize monitoring for {len(high_risk_nodes)} high-risk nodes")
            for node in high_risk_nodes[:3]:  # Top 3
                actions.append(f"Enhanced monitoring for {node['node']} (risk level: {node['risk_level']})")
        
        # General cybersecurity actions
        risk_level = risk_assessment.get("risk_level", "LOW")
        if risk_level == "HIGH":
            actions.extend([
                "Implement immediate threat containment measures",
                "Increase security team alert levels",
                "Conduct penetration testing on vulnerable nodes"
            ])
        elif risk_level == "MEDIUM":
            actions.extend([
                "Schedule security patches and updates",
                "Review and update access control policies",
                "Conduct security awareness training"
            ])
        
        return actions
    
    def _generate_pat_implementation_plan(self, pat_recommendations: Dict[str, Any]) -> List[str]:
        """Generate PAT implementation plan"""
        plan = []
        
        if not pat_recommendations or not pat_recommendations.get('recommended_pats'):
            plan.append("No specific PAT recommendations available")
            return plan
        
        best_combo = pat_recommendations['recommended_pats'][0]
        pats = best_combo.get('pats', [])
        
        plan.append(f"Implement PAT combination: {', '.join(pats)}")
        plan.append(f"Estimated cost: ${best_combo.get('total_cost', 0):,.2f}")
        plan.append(f"Implementation timeline: {best_combo.get('implementation_time', 0)} months")
        
        # Add implementation phases
        roadmap = pat_recommendations.get('implementation_roadmap', [])
        for i, step in enumerate(roadmap[:5], 1):  # First 5 steps
            plan.append(f"Phase {i}: {step.get('pat', 'N/A')} - {step.get('phase', 'Implementation')}")
        
        return plan
    
    def _generate_monitoring_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """Generate monitoring and maintenance suggestions"""
        suggestions = [
            "Implement continuous model performance monitoring",
            "Set up alerts for significant forecast deviations",
            "Regularly update models with new threat intelligence",
            "Monitor feature importance shifts over time",
            "Establish model retraining schedule based on performance decay"
        ]
        
        # Add specific suggestions based on results
        if results.get("xai_results"):
            suggestions.append("Use XAI insights to guide feature engineering improvements")
        
        if results.get("pat_recommendations"):
            suggestions.append("Monitor PAT effectiveness and adjust recommendations quarterly")
        
        return suggestions
    
    def _generate_future_improvements(self, results: Dict[str, Any]) -> List[str]:
        """Generate suggestions for future improvements"""
        improvements = [
            "Expand model ensemble with additional algorithms",
            "Incorporate external threat intelligence feeds",
            "Develop real-time forecasting capabilities",
            "Enhance XAI methods for better interpretability",
            "Implement automated model selection and hyperparameter tuning"
        ]
        
        # Identify specific areas for improvement based on results
        performance = results.get("training_results", {}).get("performance", {})
        if performance:
            avg_ori = np.mean([m.get("Operational_Readiness_Index", 0) for node in performance.values() 
                             for m in node.values()])
            if avg_ori < 0.6:
                improvements.append("Focus on improving model accuracy and operational readiness")
        
        return improvements
    
    def _create_performance_chart(self, results: Dict[str, Any], output_path: str):
        """Create model performance comparison chart"""
        try:
            model_performance = self._analyze_model_performance(results)
            ranked_models = model_performance.get("ranked_models", [])
            
            if not ranked_models:
                return
            
            models = [m["model"] for m in ranked_models]
            scores = [m["composite_score"] for m in ranked_models]
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(models, scores, color='skyblue')
            plt.xlabel('Composite Performance Score')
            plt.title('Model Performance Comparison')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', ha='right', va='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
    
    def _create_threat_forecast_chart(self, results: Dict[str, Any], output_path: str):
        """Create threat forecast overview chart"""
        try:
            forecasts = results.get("forecasts", {})
            if not forecasts:
                return
            
            # Get first few nodes for visualization
            sample_nodes = list(forecasts.keys())[:5]
            fig, axes = plt.subplots(len(sample_nodes), 1, figsize=(12, 3*len(sample_nodes)))
            
            if len(sample_nodes) == 1:
                axes = [axes]
            
            for i, node in enumerate(sample_nodes):
                model_forecasts = forecasts[node]
                for model_name, forecast_values in list(model_forecasts.items())[:3]:  # First 3 models
                    if isinstance(forecast_values, list):
                        axes[i].plot(forecast_values, label=model_name, alpha=0.7)
                
                axes[i].set_title(f'Threat Forecast - {node}')
                axes[i].set_ylabel('Threat Level')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.xlabel('Forecast Period')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating threat forecast chart: {e}")
    
    def _create_pat_recommendations_chart(self, results: Dict[str, Any], output_path: str):
        """Create PAT recommendations summary chart"""
        try:
            pat_recommendations = results.get("pat_recommendations", {})
            if not pat_recommendations or not pat_recommendations.get('recommended_pats'):
                return
            
            best_combo = pat_recommendations['recommended_pats'][0]
            pats = best_combo.get('pats', [])
            effectiveness_scores = [best_combo.get('composite_score', 0)] * len(pats)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(pats, effectiveness_scores, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'][:len(pats)])
            plt.ylabel('Effectiveness Score')
            plt.title('Recommended PATs - Effectiveness Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating PAT recommendations chart: {e}")
    
    def _prepare_results_for_llm(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for LLM processing by converting non-serializable objects"""
        def convert_obj(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_obj(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            else:
                return str(obj) if hasattr(obj, '__str__') else obj
        
        return convert_obj(results)
    
    def _generate_report_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            "report_generated": datetime.now().isoformat(),
            "report_version": "1.0",
            "total_nodes": len(results.get("forecasts", {})),
            "models_evaluated": len(set().union(*[list(node.keys()) for node in results.get("forecasts", {}).values()])),
            "xai_methods_used": len(results.get("xai_results", {})),
            "pat_recommendations_generated": bool(results.get("pat_recommendations")),
            "system_version": "EnhancedCybersecurityForecaster v4.0"
        }
    
    def _save_report_components(self, report_data: Dict[str, Any], original_results: Dict[str, Any]):
        """Save all report components to files"""
        # Save main report
        report_path = self.output_dir / "comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, cls=NumpyEncoder)
        
        # Save executive summary separately
        exec_summary_path = self.output_dir / "executive_summary.json"
        with open(exec_summary_path, 'w') as f:
            json.dump(report_data["executive_summary"], f, indent=2, cls=NumpyEncoder)
        
        # Save recommendations as text file
        rec_path = self.output_dir / "actionable_recommendations.txt"
        with open(rec_path, 'w') as f:
            f.write("ACTIONABLE RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            
            recommendations = report_data["recommendations"]
            for category, rec_list in recommendations.items():
                f.write(f"{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 30 + "\n")
                for rec in rec_list:
                    f.write(f"• {rec}\n")
                f.write("\n")
        
        # Save LLM-enhanced report if available
        if "llm_enhanced_report" in report_data:
            llm_report_path = self.output_dir / "llm_enhanced_report.md"
            with open(llm_report_path, 'w') as f:
                f.write(report_data["llm_enhanced_report"])
        
        logger.info(f"Report components saved to: {self.output_dir}")