import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage



logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for the forecasting system"""
    # LLM Configuration
    llm_provider="ollama",
    llm_model="deepseek-r1:14b",  # or "mistral", "llama2", "codellama", etc.
    base_url="http://localhost:11434" # Ollama's default URL
    api_key: Optional[str] = None
    
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
    plot_style: str = "seaborn"
    
    # Agent Configuration
    enable_agents: bool = True
    max_retries: int = 3
    
    # XAI Configuration
    enable_xai: bool = True
    xai_methods: List[str] = None
    
    # RAG Configuration
    enable_rag: bool = True
    rag_db_path: str = "cyber_threat_db"
    
    # New Models
    enable_lstm: bool = True
    enable_prophet: bool = True
    enable_validation: bool = True  
    validation_config: Dict[str, Any] = { #change
            'n_candidates': 3,
            'k_models': 3, 
            'cv_folds': 5,
            'optimization_method': 'grid_search'
        },
    
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
        if self.config.llm_provider == "openai":
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
            )
        elif self.config.llm_provider == "ollama":
            return Ollama(
                model=self.config.llm_model,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the configured LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

@dataclass
class PATConfig:
    """Configuration for PAT Recommendation Agent"""
    graph_path: str = "graph.csv"
    recommendation_threshold: float = 0.7
    max_recommendations: int = 5
    cost_optimization: bool = True
    effectiveness_weight: float = 0.6
    cost_weight: float = 0.3
    implementation_weight: float = 0.1
    enable_llm_optimization: bool = True

class PATRecommendationAgent:
    """
    Agent for recommending Pertinent Alleviation Technologies (PATs)
    based on forecasted cyber threats and graph relationships
    """
    
    def __init__(self, config: PATConfig, llm_manager=None):
        self.config = config
        self.llm_manager = llm_manager
        self.pat_graph = None
        self.attack_pat_mapping = None
        self.pat_metadata = None
        
        # Load PAT graph data
        self._load_pat_graph()
    
    def _load_pat_graph(self):
        """Load PAT graph data from CSV file"""
        try:
            logger.info("Loading PAT graph data...")
            self.pat_graph = pd.read_csv(self.config.graph_path, index_col=0)
            
            # Extract attack names (row indices) and PATs (columns)
            self.attack_names = self.pat_graph.index.tolist()
            self.pat_names = self.pat_graph.columns.tolist()
            
            logger.info(f"Loaded PAT graph with {len(self.attack_names)} attacks and {len(self.pat_names)} PATs")
            
            # Create normalized mapping (0-1 effectiveness scores)
            self.attack_pat_mapping = self.pat_graph.to_dict()
            
            # Initialize PAT metadata (could be extended from external source)
            self._initialize_pat_metadata()
            
        except Exception as e:
            logger.error(f"Error loading PAT graph: {e}")
            raise
    
    def _initialize_pat_metadata(self):
        """Initialize PAT metadata with cost, implementation complexity, etc."""
        self.pat_metadata = {}
        
        for pat in self.pat_names:
            # Default metadata - in practice, this would come from a database
            self.pat_metadata[pat] = {
                'estimated_cost': np.random.uniform(1000, 50000),  # USD
                'implementation_time': np.random.uniform(1, 12),   # months
                'complexity': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                'coverage': np.random.uniform(0.5, 1.0),  # Overall threat coverage
                'maintenance_required': np.random.choice([True, False], p=[0.7, 0.3])
            }
    
    def recommend_pats_for_threats(self, 
                                 forecasted_threats: Dict[str, float],
                                 budget_constraint: Optional[float] = None,
                                 time_constraint: Optional[float] = None) -> Dict[str, Any]:
        """
        Recommend PATs for forecasted threats
        
        Args:
            forecasted_threats: Dictionary of {threat_name: probability_score}
            budget_constraint: Optional maximum budget
            time_constraint: Optional maximum implementation time
            
        Returns:
            Dictionary with PAT recommendations and rationale
        """
        logger.info(f"Generating PAT recommendations for {len(forecasted_threats)} forecasted threats")
        
        # Step 1: Identify relevant PATs for each threat
        threat_pat_mapping = self._map_threats_to_pats(forecasted_threats)
        
        # Step 2: Score and rank PATs
        pat_scores = self._score_pats(threat_pat_mapping, forecasted_threats)
        
        # Step 3: Apply constraints
        if budget_constraint or time_constraint:
            pat_scores = self._apply_constraints(pat_scores, budget_constraint, time_constraint)
        
        # Step 4: Generate optimal PAT combinations
        optimal_combinations = self._generate_optimal_combinations(pat_scores, forecasted_threats)
        
        # Step 5: LLM-enhanced optimization (if enabled)
        if self.config.enable_llm_optimization and self.llm_manager:
            optimized_recommendations = self._llm_optimize_recommendations(
                optimal_combinations, forecasted_threats, budget_constraint, time_constraint
            )
        else:
            optimized_recommendations = optimal_combinations
        
        return {
            'threat_analysis': forecasted_threats,
            'recommended_pats': optimized_recommendations,
            'coverage_analysis': self._calculate_coverage_analysis(optimized_recommendations, forecasted_threats),
            'risk_reduction_metrics': self._calculate_risk_reduction(optimized_recommendations, forecasted_threats),
            'implementation_roadmap': self._generate_implementation_roadmap(optimized_recommendations)
        }
    
    def _map_threats_to_pats(self, forecasted_threats: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
        """Map forecasted threats to relevant PATs with effectiveness scores"""
        threat_pat_mapping = {}
        
        for threat, probability in forecasted_threats.items():
            if threat in self.attack_pat_mapping:
                relevant_pats = []
                
                for pat, effectiveness in self.attack_pat_mapping[threat].items():
                    if effectiveness >= self.config.recommendation_threshold:
                        # Adjust score based on threat probability
                        adjusted_score = effectiveness * probability
                        relevant_pats.append((pat, adjusted_score, effectiveness))
                
                # Sort by adjusted score
                relevant_pats.sort(key=lambda x: x[1], reverse=True)
                threat_pat_mapping[threat] = relevant_pats[:self.config.max_recommendations]
        
        return threat_pat_mapping
    
    def _score_pats(self, threat_pat_mapping: Dict[str, List[Tuple[str, float]]], 
                   forecasted_threats: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Score PATs based on multiple criteria"""
        pat_scores = {}
        
        # Collect all PATs across all threats
        all_pats = set()
        for threat_pats in threat_pat_mapping.values():
            for pat_data in threat_pats:
                all_pats.add(pat_data[0])
        
        # Score each PAT
        for pat in all_pats:
            effectiveness_score = 0
            covered_threats = []
            total_threat_probability = 0
            
            # Calculate effectiveness across all covered threats
            for threat, probability in forecasted_threats.items():
                if threat in threat_pat_mapping:
                    for pat_data in threat_pat_mapping[threat]:
                        if pat_data[0] == pat:
                            effectiveness_score += pat_data[1]  # Adjusted score
                            covered_threats.append({
                                'threat': threat,
                                'probability': probability,
                                'effectiveness': pat_data[2]
                            })
                            total_threat_probability += probability
            
            if covered_threats:
                # Normalize effectiveness score
                normalized_effectiveness = effectiveness_score / len(forecasted_threats) if forecasted_threats else 0
                
                # Get PAT metadata
                metadata = self.pat_metadata.get(pat, {})
                
                # Calculate composite score
                cost_score = 1 - (metadata.get('estimated_cost', 0) / 50000)  # Normalize cost
                implementation_score = 1 - (metadata.get('implementation_time', 12) / 12)  # Normalize time
                
                composite_score = (
                    self.config.effectiveness_weight * normalized_effectiveness +
                    self.config.cost_weight * cost_score +
                    self.config.implementation_weight * implementation_score
                )
                
                pat_scores[pat] = {
                    'composite_score': composite_score,
                    'effectiveness_score': normalized_effectiveness,
                    'cost_score': cost_score,
                    'implementation_score': implementation_score,
                    'covered_threats': covered_threats,
                    'threat_coverage': len(covered_threats) / len(forecasted_threats) if forecasted_threats else 0,
                    'total_threat_probability': total_threat_probability,
                    'metadata': metadata
                }
        
        return pat_scores
    
    def _apply_constraints(self, pat_scores: Dict[str, Dict[str, Any]], 
                         budget_constraint: Optional[float], 
                         time_constraint: Optional[float]) -> Dict[str, Dict[str, Any]]:
        """Apply budget and time constraints to PAT recommendations"""
        constrained_scores = pat_scores.copy()
        
        for pat, scores in list(constrained_scores.items()):
            metadata = scores['metadata']
            
            # Apply budget constraint
            if budget_constraint and metadata.get('estimated_cost', 0) > budget_constraint:
                del constrained_scores[pat]
                continue
            
            # Apply time constraint
            if time_constraint and metadata.get('implementation_time', 12) > time_constraint:
                del constrained_scores[pat]
                continue
        
        return constrained_scores
    
    def _generate_optimal_combinations(self, pat_scores: Dict[str, Dict[str, Any]],
                                    forecasted_threats: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate optimal combinations of PATs"""
        if not pat_scores:
            return []
        
        # Sort PATs by composite score
        sorted_pats = sorted(pat_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        combinations = []
        
        # Single PAT recommendations
        for i, (pat, scores) in enumerate(sorted_pats[:5]):
            combinations.append({
                'combination_id': f"single_{i+1}",
                'pats': [pat],
                'composite_score': scores['composite_score'],
                'total_cost': scores['metadata']['estimated_cost'],
                'implementation_time': scores['metadata']['implementation_time'],
                'threat_coverage': scores['threat_coverage'],
                'covered_threats': scores['covered_threats']
            })
        
        # Two-PAT combinations (top combinations)
        for i in range(min(3, len(sorted_pats))):
            for j in range(i+1, min(6, len(sorted_pats))):
                pat1, scores1 = sorted_pats[i]
                pat2, scores2 = sorted_pats[j]
                
                # Calculate combined metrics
                combined_threats = self._combine_covered_threats(
                    scores1['covered_threats'], scores2['covered_threats']
                )
                
                combined_coverage = len(combined_threats) / len(forecasted_threats) if forecasted_threats else 0
                total_cost = scores1['metadata']['estimated_cost'] + scores2['metadata']['estimated_cost']
                max_implementation_time = max(
                    scores1['metadata']['implementation_time'],
                    scores2['metadata']['implementation_time']
                )
                
                # Combined score (prioritizing coverage and cost efficiency)
                combined_score = (
                    0.6 * combined_coverage +
                    0.2 * (1 - total_cost / 100000) +  # Normalize cost
                    0.2 * (1 - max_implementation_time / 24)  # Normalize time
                )
                
                combinations.append({
                    'combination_id': f"combo_{len(combinations) + 1}",
                    'pats': [pat1, pat2],
                    'composite_score': combined_score,
                    'total_cost': total_cost,
                    'implementation_time': max_implementation_time,
                    'threat_coverage': combined_coverage,
                    'covered_threats': combined_threats
                })
        
        # Sort all combinations by composite score
        combinations.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return combinations[:self.config.max_recommendations]
    
    def _combine_covered_threats(self, threats1: List[Dict], threats2: List[Dict]) -> List[Dict]:
        """Combine covered threats from multiple PATs"""
        combined = {}
        
        for threat_list in [threats1, threats2]:
            for threat_data in threat_list:
                threat_name = threat_data['threat']
                if threat_name not in combined:
                    combined[threat_name] = threat_data
                else:
                    # Keep the higher effectiveness
                    if threat_data['effectiveness'] > combined[threat_name]['effectiveness']:
                        combined[threat_name] = threat_data
        
        return list(combined.values())
    
    def _llm_optimize_recommendations(self, combinations: List[Dict[str, Any]],
                                    forecasted_threats: Dict[str, float],
                                    budget_constraint: Optional[float],
                                    time_constraint: Optional[float]) -> List[Dict[str, Any]]:
        """Use LLM to optimize and provide rationale for PAT recommendations"""
        if not self.llm_manager:
            return combinations
        
        system_prompt = """You are a cybersecurity expert specializing in threat mitigation technology recommendations. 
        Analyze the PAT combinations and provide optimized recommendations with detailed rationale."""
        
        user_prompt = f"""
        Given the following forecasted cyber threats and PAT combinations, provide optimized recommendations:
        
        Forecasted Threats: {json.dumps(forecasted_threats, indent=2)}
        
        PAT Combinations: {json.dumps(combinations[:3], indent=2, default=str)}
        
        Constraints:
        - Budget: {budget_constraint if budget_constraint else 'Unconstrained'}
        - Time: {time_constraint if time_constraint else 'Unconstrained'}
        
        Please provide:
        1. Recommended PAT combination with rationale
        2. Risk reduction analysis
        3. Implementation considerations
        4. Cost-benefit analysis
        
        Return JSON format:
        {{
            "optimized_recommendations": [
                {{
                    "combination_id": "str",
                    "rationale": "str",
                    "expected_risk_reduction": float,
                    "key_benefits": ["str"],
                    "implementation_considerations": ["str"]
                }}
            ]
        }}
        """
        
        try:
            llm_response = self.llm_manager.generate_response(system_prompt, user_prompt)
            optimized_data = self._parse_llm_response(llm_response)
            
            # Merge LLM insights with original combinations
            for combo in combinations:
                for optimized in optimized_data.get('optimized_recommendations', []):
                    if combo['combination_id'] == optimized['combination_id']:
                        combo.update(optimized)
            
            return combinations
            
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            return combinations
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for PAT recommendations"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {'optimized_recommendations': []}
    
    def _calculate_coverage_analysis(self, recommendations: List[Dict[str, Any]],
                                  forecasted_threats: Dict[str, float]) -> Dict[str, Any]:
        """Calculate coverage analysis for recommended PATs"""
        if not recommendations:
            return {}
        
        best_combo = recommendations[0]
        covered_threat_names = [t['threat'] for t in best_combo.get('covered_threats', [])]
        
        return {
            'total_threats': len(forecasted_threats),
            'covered_threats': len(covered_threat_names),
            'coverage_percentage': best_combo.get('threat_coverage', 0) * 100,
            'high_risk_coverage': self._calculate_high_risk_coverage(covered_threat_names, forecasted_threats),
            'critical_gaps': [t for t in forecasted_threats if t not in covered_threat_names]
        }
    
    def _calculate_high_risk_coverage(self, covered_threats: List[str], 
                                    forecasted_threats: Dict[str, float]) -> float:
        """Calculate coverage of high-risk threats (probability > 0.7)"""
        high_risk_threats = [t for t, p in forecasted_threats.items() if p > 0.7]
        if not high_risk_threats:
            return 0.0
        
        covered_high_risk = [t for t in covered_threats if t in high_risk_threats]
        return len(covered_high_risk) / len(high_risk_threats)
    
    def _calculate_risk_reduction(self, recommendations: List[Dict[str, Any]],
                                forecasted_threats: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk reduction metrics"""
        if not recommendations:
            return {}
        
        best_combo = recommendations[0]
        
        total_risk = sum(forecasted_threats.values())
        residual_risk = total_risk
        
        # Calculate residual risk after PAT implementation
        for threat, probability in forecasted_threats.items():
            max_effectiveness = 0
            for pat_data in best_combo.get('covered_threats', []):
                if pat_data['threat'] == threat:
                    max_effectiveness = max(max_effectiveness, pat_data['effectiveness'])
            
            residual_risk -= probability * max_effectiveness
        
        risk_reduction = (total_risk - residual_risk) / total_risk if total_risk > 0 else 0
        
        return {
            'initial_risk_score': total_risk,
            'residual_risk_score': max(0, residual_risk),
            'risk_reduction_percentage': risk_reduction * 100,
            'risk_reduction_efficiency': risk_reduction / best_combo.get('total_cost', 1) * 1000
        }
    
    def _generate_implementation_roadmap(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate implementation roadmap for recommended PATs"""
        if not recommendations:
            return []
        
        roadmap = []
        best_combo = recommendations[0]
        
        for pat in best_combo.get('pats', []):
            metadata = self.pat_metadata.get(pat, {})
            roadmap.append({
                'pat': pat,
                'phase': 'Immediate' if metadata.get('implementation_time', 12) <= 3 else 'Short-term',
                'estimated_timeline': f"{metadata.get('implementation_time', 0)} months",
                'required_resources': self._estimate_required_resources(pat),
                'dependencies': [],
                'success_metrics': self._define_success_metrics(pat)
            })
        
        return roadmap
    
    def _estimate_required_resources(self, pat: str) -> Dict[str, Any]:
        """Estimate resources required for PAT implementation"""
        # This would typically come from a knowledge base
        return {
            'team_size': np.random.randint(1, 5),
            'expertise_required': ['Cybersecurity', 'Network Engineering'],
            'tools_required': [],
            'training_required': True
        }
    
    def _define_success_metrics(self, pat: str) -> List[str]:
        """Define success metrics for PAT implementation"""
        return [
            f"Reduction in {pat}-related incidents",
            "Improved detection capabilities",
            "Enhanced response times"
        ]
    
    def visualize_recommendations(self, recommendations: Dict[str, Any], 
                                save_path: Optional[str] = None):
        """Visualize PAT recommendations"""
        if not recommendations:
            logger.warning("No recommendations to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PAT Recommendation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: PAT Effectiveness Comparison
        self._plot_effectiveness_comparison(axes[0, 0], recommendations)
        
        # Plot 2: Threat Coverage Analysis
        self._plot_coverage_analysis(axes[0, 1], recommendations)
        
        # Plot 3: Cost-Benefit Analysis
        self._plot_cost_benefit_analysis(axes[1, 0], recommendations)
        
        # Plot 4: Risk Reduction Impact
        self._plot_risk_reduction(axes[1, 1], recommendations)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PAT visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_effectiveness_comparison(self, ax, recommendations: Dict[str, Any]):
        """Plot PAT effectiveness comparison"""
        combos = recommendations.get('recommended_pats', [])
        if not combos:
            return
        
        combo_names = [combo['combination_id'] for combo in combos]
        scores = [combo['composite_score'] for combo in combos]
        
        bars = ax.bar(combo_names, scores, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'])
        ax.set_title('PAT Combination Effectiveness Scores')
        ax.set_ylabel('Composite Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    def _plot_coverage_analysis(self, ax, recommendations: Dict[str, Any]):
        """Plot threat coverage analysis"""
        coverage_analysis = recommendations.get('coverage_analysis', {})
        if not coverage_analysis:
            return
        
        labels = ['Covered Threats', 'Uncovered Threats']
        sizes = [
            coverage_analysis.get('covered_threats', 0),
            coverage_analysis.get('total_threats', 0) - coverage_analysis.get('covered_threats', 0)
        ]
        colors = ['#2E8B57', '#FF6347']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Threat Coverage Analysis')
    
    def _plot_cost_benefit_analysis(self, ax, recommendations: Dict[str, Any]):
        """Plot cost-benefit analysis"""
        combos = recommendations.get('recommended_pats', [])
        if not combos:
            return
        
        costs = [combo.get('total_cost', 0) for combo in combos]
        benefits = [combo.get('composite_score', 0) * 100 for combo in combos]  # Scale for visibility
        combo_names = [combo['combination_id'] for combo in combos]
        
        x = np.arange(len(combo_names))
        width = 0.35
        
        ax.bar(x - width/2, costs, width, label='Cost (USD)', color='#FF6347', alpha=0.7)
        ax.bar(x + width/2, benefits, width, label='Benefit Score', color='#2E8B57', alpha=0.7)
        
        ax.set_xlabel('PAT Combinations')
        ax.set_ylabel('Cost vs Benefit')
        ax.set_title('Cost-Benefit Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(combo_names, rotation=45)
        ax.legend()
    
    def _plot_risk_reduction(self, ax, recommendations: Dict[str, Any]):
        """Plot risk reduction impact"""
        risk_metrics = recommendations.get('risk_reduction_metrics', {})
        if not risk_metrics:
            return
        
        labels = ['Initial Risk', 'Residual Risk']
        values = [
            risk_metrics.get('initial_risk_score', 0),
            risk_metrics.get('residual_risk_score', 0)
        ]
        
        bars = ax.bar(labels, values, color=['#FF6347', '#2E8B57'])
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Reduction Impact')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Add reduction percentage
        reduction = risk_metrics.get('risk_reduction_percentage', 0)
        ax.text(0.5, max(values) * 0.8, f'Risk Reduction: {reduction:.1f}%', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

