import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

PREPROCESS_SYSTEM_PROMPT = """
You are the Data Preprocessing Chief Agent for an advanced time series forecasting system. 
Your mission is to ensure that all input data is of the highest possible quality before it enters the modeling pipeline.

Background:
- You have deep expertise in time series data cleaning, anomaly detection, and preparation for machine learning and statistical forecasting.
- You understand the downstream impact of preprocessing choices on model performance and interpretability.

Your responsibilities:
- Rigorously assess the quality of the input time series, identifying missing values, outliers, and structural issues.
- For each issue, recommend the most appropriate handling strategy, considering both statistical best practices and the needs of advanced forecasting models.
- Justify your recommendations with clear reasoning, referencing both the data characteristics and potential modeling implications.
- If relevant, suggest additional preprocessing steps (e.g., resampling, detrending, feature engineering) that could improve results.
- Always return your decisions in a structured Python dict, and ensure your reasoning is transparent and actionable.

You have access to:
- The raw time series data (as a Python dict)
- Any prior preprocessing history or known data issues

Your output will directly determine how the data is prepared for all subsequent analysis and modeling.
"""

class DataLoader:
    """Data loading utilities"""
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load CSV file"""
        return pd.read_csv(file_path)
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate DataFrame structure"""
        return isinstance(df, pd.DataFrame) and not df.empty

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values using specified strategy"""
        df = data.copy()
        
        if strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif strategy == 'forward_fill':
            df = df.ffill()
        elif strategy == 'backward_fill':
            df = df.bfill()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'zero':
            df = df.fillna(0)
        elif strategy == 'drop':
            df = df.dropna()
        
        return df
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using specified method"""
        outlier_info = {}
        
        for col in data.columns:
            series = data[col].dropna()
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(series))
                outliers = series[z_scores > threshold]
                
            elif method == 'percentile':
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outlier_info[col] = {
                'outliers': outliers.tolist(),
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100 if len(series) > 0 else 0
            }
        
        return outlier_info
    
    def handle_outliers(self, data: pd.DataFrame, outlier_info: Dict[str, Any], strategy: str = 'clip') -> pd.DataFrame:
        """Handle outliers using specified strategy"""
        df = data.copy()
        
        for col, info in outlier_info.items():
            if col in df.columns and info['outliers']:
                outliers = info['outliers']
                
                if strategy == 'clip':
                    # Clip to min/max of non-outlier range
                    non_outliers = df[col][~df[col].isin(outliers)]
                    if len(non_outliers) > 0:
                        lower_bound = non_outliers.min()
                        upper_bound = non_outliers.max()
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif strategy == 'interpolate':
                    # Mark outliers as NaN and interpolate
                    outlier_mask = df[col].isin(outliers)
                    df.loc[outlier_mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                
                elif strategy in ['mean', 'median']:
                    # Replace with mean/median
                    replacement = df[col].mean() if strategy == 'mean' else df[col].median()
                    outlier_mask = df[col].isin(outliers)
                    df.loc[outlier_mask, col] = replacement
        
        return df

class DataValidator:
    """Data validation utilities"""
    
    def validate_time_series(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate time series data"""
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append("DataFrame is empty")
        
        # Check for infinite values
        if np.any(np.isinf(data.select_dtypes(include=[np.number]))):
            warnings.append("Data contains infinite values")
        
        # Check data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            warnings.append("No numeric columns found")
        
        # Check for constant columns
        for col in numeric_cols:
            if data[col].std() == 0:
                warnings.append(f"Column '{col}' has zero variance")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'numeric_columns': list(numeric_cols),
            'total_rows': len(data),
            'total_columns': len(data.columns)
        }

class TimeSeriesVisualizer:
    """Time series visualization utilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        plt.style.use('seaborn-v0_8')
    
    def create_time_series_plot(self, data: pd.DataFrame, title: str = "Time Series Plot") -> plt.Figure:
        """Create basic time series plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_distribution_plot(self, data: pd.DataFrame, title: str = "Distribution Analysis") -> plt.Figure:
        """Create distribution analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        for col in data.columns:
            axes[0, 0].plot(data.index, data[col], label=col, linewidth=2)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram with KDE
        for col in data.columns:
            sns.histplot(data[col].dropna(), kde=True, ax=axes[0, 1], label=col, alpha=0.6)
        axes[0, 1].set_title('Value Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Box plot
        data_to_plot = [data[col].dropna() for col in data.columns]
        axes[1, 0].boxplot(data_to_plot, labels=data.columns)
        axes[1, 0].set_title('Value Box Plot')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot (for first column)
        if len(data.columns) > 0:
            first_col = data.columns[0]
            stats.probplot(data[first_col].dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f'Q-Q Plot - {first_col}')
        
        plt.tight_layout()
        return fig
    
    def create_rolling_stats_plot(self, data: pd.DataFrame, window: int = 24, 
                                title: str = "Rolling Statistics") -> plt.Figure:
        """Create rolling statistics plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in data.columns:
            rolling_mean = data[col].rolling(window=window).mean()
            rolling_std = data[col].rolling(window=window).std()
            
            ax.plot(data.index, rolling_mean, label=f'{col} - Rolling Mean', linewidth=2)
            ax.plot(data.index, rolling_std, label=f'{col} - Rolling Std', alpha=0.7, linewidth=2)
        
        ax.set_title(f'{title} (Window={window})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_autocorrelation_plot(self, data: pd.DataFrame, title: str = "Autocorrelation Analysis") -> plt.Figure:
        """Create autocorrelation and partial autocorrelation plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 0:
                plot_acf(series, ax=axes[0], lags=min(40, len(series)//2), title='Autocorrelation')
                plot_pacf(series, ax=axes[1], lags=min(40, len(series)//2), title='Partial Autocorrelation')
        
        plt.tight_layout()
        return fig
    
    def create_seasonal_decomposition_plot(self, data: pd.DataFrame, period: int = 12,
                                         title: str = "Seasonal Decomposition") -> plt.Figure:
        """Create seasonal decomposition plot"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        for col in data.columns:
            series = data[col].dropna()
            if len(series) >= 2 * period:
                try:
                    decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
                    
                    decomposition.observed.plot(ax=axes[0], title='Observed', linewidth=2)
                    decomposition.trend.plot(ax=axes[1], title='Trend', linewidth=2)
                    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', linewidth=2)
                    decomposition.resid.plot(ax=axes[3], title='Residual', linewidth=2)
                except Exception as e:
                    logger.warning(f"Seasonal decomposition failed for {col}: {e}")
        
        plt.tight_layout()
        return fig

class FileSaver:
    """File saving utilities"""
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data as JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def save_dataframe(data: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Save DataFrame to CSV"""
        data.to_csv(file_path, index=True)

class ExperimentMemory:
    """Simple experiment memory management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.storage = {}
        self.history = []
    
    def store(self, key: str, value: Any, category: str = 'default') -> None:
        """Store value in memory"""
        self.storage[f"{category}_{key}"] = value
    
    def retrieve(self, key: str, category: str = 'default') -> Any:
        """Retrieve value from memory"""
        return self.storage.get(f"{category}_{key}")
    
    def add_history(self, action: str, details: Dict[str, Any]) -> None:
        """Add action to history"""
        self.history.append({
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

class PreprocessLLMTools:
    def __init__(self, llm):
        self.llm = llm

    def get_preprocess_decision_prompt(self, data: pd.DataFrame) -> str:
        return f"""
You are a time series data preprocessing expert.

Given the following time series data (as a Python dict):

{data.to_dict(orient='list')}

Please:
1. Assess the overall data quality.
2. Recommend a missing value handling strategy (choose from: interpolate, forward_fill, backward_fill, mean, median, drop, zero).
3. Recommend an outlier handling strategy (choose from: clip, drop, zero, interpolate, ffill, bfill, mean, median, smooth).
4. Optionally, suggest any other preprocessing steps if needed.

Return your answer as a Python dict:
{
  "quality_assessment": "string",
  "missing_value_strategy": "string",
  "outlier_strategy": "string",
  "other_suggestions": "string"
}
"""

    def analyze_data_quality(self, data: pd.DataFrame) -> dict:
        prompt = self.get_preprocess_decision_prompt(data)
        response = self.llm.invoke([SystemMessage(content=PREPROCESS_SYSTEM_PROMPT), 
                                    HumanMessage(content=prompt)])
        return response.content

class PreprocessAgent:
    """
    Data preprocessing Agent
    Responsible for data loading, cleaning, validation, and visualization.
    """
    
    def __init__(self, model: str = "gpt-4", config: dict = None):
        """
        Initialize the preprocessing agent.
        """
        try:
            from langchain_community.llms import Ollama
            self.llm = Ollama (
                model="deepseek-r1:8b",
                base_url="http://localhost:11434",
                temperature=0.1
            )
        except ImportError:
            logger.warning("ollama not available, LLM functionality disabled")
            self.llm = None
            
        self.tools = PreprocessLLMTools(self.llm) if self.llm else None
        self.config = config or {}
        self.visualizer = TimeSeriesVisualizer(self.config)
        
        # Get preprocessing configuration
        self.preprocess_config = self.config.get('preprocess', {})
        self.outlier_threshold = self.preprocess_config.get('outlier_threshold', 1.5)
        
        logger.info("PreprocessAgent initialized")
        self.memory = ExperimentMemory(self.config)
    
    def process(self, data: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Execute the complete preprocessing workflow.
        """
        logger.info("Starting data preprocessing...")
        
        # Debug: Print data index information
        logger.info(f"Input data index range: {data.index.min()} to {data.index.max()}")
        
        try:
            # 1. Data validation
            validation_result = self._validate_data(data)
            
            # 2. Initial data quality analysis for preprocessing strategies
            initial_quality_analysis = self._analyze_data_quality(data, {})
            
            # Extract recommended strategies
            missing_value_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('missing_value_strategy', 'interpolate')
            outlier_handle_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('outlier_handle_strategy', 'clip')
            outlier_detect_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('outlier_detect_strategy', 'iqr')
            
            logger.info(f"Recommended missing value strategy: {missing_value_strategy}")
            logger.info(f"Recommended outlier detect strategy: {outlier_detect_strategy}")
            logger.info(f"Recommended outlier handle strategy: {outlier_handle_strategy}")
            
            # 3. Clean data using recommended strategy
            cleaned_data = self._clean_data(data, missing_value_strategy)
            
            # 4. Detect outliers
            outlier_info = self._detect_outliers(cleaned_data, outlier_detect_strategy)
            
            # 5. Handle outliers using recommended strategy
            if outlier_info:
                cleaned_data = self._handle_outliers(cleaned_data, outlier_info, outlier_handle_strategy)
            
            # 6. Generate visualizations
            visualizations = self._generate_visualizations(cleaned_data, output_dir)
            
            # 7. Generate comprehensive analysis report
            analysis_report = self._generate_comprehensive_analysis_report(cleaned_data, visualizations)
            
            # 8. Save preprocessing results
            self._save_preprocess_results(cleaned_data, analysis_report, output_dir)
            
            # 9. Update memory
            self._update_memory(cleaned_data, analysis_report, visualizations)
            
            result = {
                'cleaned_data': cleaned_data,
                'analysis_report': analysis_report,
                'outlier_info': outlier_info,
                'validation_result': validation_result,
                'visualizations': visualizations,
                'preprocess_config': {
                    'missing_strategy': missing_value_strategy,
                    'outlier_detect_strategy': outlier_detect_strategy,
                    'outlier_handle_strategy': outlier_handle_strategy,
                }
            }
            
            logger.info("Data preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def run(self, data: pd.DataFrame, output_dir: str = None):
        """Run the preprocessing agent"""
        import time
        time.sleep(0.5)
        return self.process(data, output_dir)
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data"""
        logger.info("Validating data...")
        validator = DataValidator()
        validation_result = validator.validate_time_series(data)
        
        if not validation_result['is_valid']:
            logger.warning(f"Data validation issues found: {validation_result['errors']}")
        else:
            logger.info("Data validation passed")
        
        return validation_result
    
    def _clean_data(self, data: pd.DataFrame, missing_strategy: str) -> pd.DataFrame:
        """Clean data"""
        logger.info(f"Cleaning data with strategy: {missing_strategy}")
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(data, strategy=missing_strategy)
        
        missing_count = cleaned_data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Still have {missing_count} missing values after cleaning")
        else:
            logger.info("All missing values handled successfully")
        
        return cleaned_data
    
    def _detect_outliers(self, data: pd.DataFrame, outlier_strategy: str) -> Dict[str, Any]:
        """Detect outliers in the data"""
        logger.info("Detecting outliers...")
        preprocessor = DataPreprocessor()
        outlier_info = preprocessor.detect_outliers(
            data, 
            method=outlier_strategy,
            threshold=self.outlier_threshold
        )
        
        if outlier_info and any(info['count'] > 0 for info in outlier_info.values()):
            logger.info(f"Found outliers in columns: {[col for col, info in outlier_info.items() if info['count'] > 0]}")
            return outlier_info
        else:
            logger.info("No outliers detected")
            return {}
    
    def _handle_outliers(self, data: pd.DataFrame, outlier_info: Dict[str, Any], outlier_strategy: str) -> pd.DataFrame:
        """Handle outliers based on detected outliers"""
        logger.info(f"Handling outliers with strategy: {outlier_strategy}")
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_outliers(data, outlier_info, strategy=outlier_strategy)
        return cleaned_data
    
    def _generate_comprehensive_analysis_report(self, data: pd.DataFrame, visualizations: Dict[str, str]) -> Dict[str, Any]:
        """Analyze data quality and recommend preprocessing strategies"""
        logger.info("Analyzing data quality...")
        
        if not self.llm:
            return self._generate_fallback_analysis_report(data)
        
        sample = data.to_dict(orient='list')
        prompt = f"""
Given the following preprocessed time series data and generated visualizations, please provide a comprehensive analysis report.

Data (as a Python dict):
{sample}

Generated Visualizations:
{visualizations}

Note: This data has already been preprocessed - missing values and outliers have been handled.

Please provide a comprehensive analysis including:

1. Data Overview:
   - basic_stats: mean, std, min, max, trend
   - data_characteristics: seasonality, stationarity, patterns

2. Data Quality Assessment:
   - data_quality_score: overall quality score (0-1) after preprocessing
   - data_characteristics: key characteristics of the cleaned data

3. Insights from Visualizations:
   - key_patterns: patterns observed in the data
   - seasonal_components: any seasonal patterns
   - trend_analysis: overall trend direction and strength
   - distribution_characteristics: data distribution insights

4. Forecasting Readiness:
   - data_suitability: how suitable this data is for forecasting
   - potential_challenges: any challenges for forecasting models
   - data_strengths: strengths of this dataset

5. Model and Feature Recommendations:
   - model_suggestions: suitable model types for this data
   - feature_engineering: suggested features to create
   - preprocessing_effectiveness: how well the preprocessing worked

IMPORTANT: Return ONLY the JSON object below, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON.

{{
    "data_overview": {{
        "basic_stats": {{
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "trend": "string"
        }},
        "data_characteristics": {{
            "seasonality": "string",
            "stationarity": "string",
            "patterns": ["string"]
        }}
    }},
    "quality_assessment": {{
        "data_quality_score": float,
        "data_characteristics": "string"
    }},
    "visualization_insights": {{
        "key_patterns": ["string"],
        "seasonal_components": "string",
        "trend_analysis": "string",
        "distribution_characteristics": "string"
    }},
    "forecasting_readiness": {{
        "data_suitability": "string",
        "potential_challenges": ["string"],
        "data_strengths": ["string"]
    }},
    "recommendations": {{
        "model_suggestions": ["string"],
        "feature_engineering": ["string"],
        "preprocessing_effectiveness": "string"
    }}
}}
"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            if not response.content or response.content.strip() == "":
                logger.warning("LLM returned empty response, using fallback analysis report")
                return self._generate_fallback_analysis_report(data)
            
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.warning("JSON parsing failed, using fallback analysis report")
            return self._generate_fallback_analysis_report(data)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._generate_fallback_analysis_report(data)
    
    def _generate_fallback_analysis_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate fallback analysis report when LLM fails"""
        logger.info("Generating fallback analysis report...")
        
        basic_stats = {
            "mean": float(data.mean().iloc[0]) if not data.empty else 0.0,
            "std": float(data.std().iloc[0]) if not data.empty else 0.0,
            "min": float(data.min().iloc[0]) if not data.empty else 0.0,
            "max": float(data.max().iloc[0]) if not data.empty else 0.0,
            "trend": "stable"
        }
        
        if len(data) > 1:
            try:
                slope = np.polyfit(range(len(data)), data.iloc[:, 0], 1)[0]
                if slope > 0.01:
                    basic_stats["trend"] = "increasing"
                elif slope < -0.01:
                    basic_stats["trend"] = "decreasing"
            except:
                pass
        
        return {
            "data_overview": {
                "basic_stats": basic_stats,
                "data_characteristics": {
                    "seasonality": "unknown",
                    "stationarity": "unknown",
                    "patterns": ["data_loaded_successfully"]
                }
            },
            "quality_assessment": {
                "data_quality_score": 0.8,
                "data_characteristics": "Data has been preprocessed and is ready for analysis"
            },
            "visualization_insights": {
                "key_patterns": ["data_available_for_analysis"],
                "seasonal_components": "unknown",
                "trend_analysis": f"Overall trend appears to be {basic_stats['trend']}",
                "distribution_characteristics": "Data distribution available for analysis"
            },
            "forecasting_readiness": {
                "data_suitability": "suitable",
                "potential_challenges": ["limited_insights_due_to_llm_failure"],
                "data_strengths": ["preprocessed_data", "basic_statistics_available"]
            },
            "recommendations": {
                "model_suggestions": ["ARIMA", "ExponentialSmoothing", "LinearRegression"],
                "feature_engineering": ["lag_features", "rolling_statistics"],
                "preprocessing_effectiveness": "preprocessing_completed_successfully"
            }
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame, outlier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data quality and recommend preprocessing strategies"""
        logger.info("Analyzing data quality for preprocessing strategies...")
        
        if not self.llm:
            return self._generate_fallback_data_quality_analysis(data, outlier_info)
        
        sample = data.to_dict(orient='list')
        prompt = f"""
Given the following time series data (as a Python dict):

{sample}

Please analyze the data quality and provide the following information as a JSON file:

1. Basic statistics for each column:
   - mean: float
   - std: float  
   - min: float
   - max: float
   - trend: 'increasing'/'decreasing'/'stable'

2. Missing value information:
   - missing_count: int (total missing values)
   - missing_percentage: float (percentage of missing values)

3. Outlier information:
   - outlier_count: int (total outliers detected)
   - outlier_percentage: float (percentage of outliers in the data, between 0 and 1)   

4. Data quality assessment:
   - data_quality_score: float (0-1, where 1 is perfect quality)
   - main_issues: list of strings (e.g., ['missing_values', 'outliers', 'noise', ...])

5. Recommended preprocessing strategies:
   - missing_value_strategy: string (choose from: 'interpolate', 'forward_fill', 'backward_fill', 'mean', 'median', 'drop', 'zero')
   - outlier_detect_strategy: string (choose from: 'iqr', 'zscore', 'percentile', 'none')
   - outlier_handle_strategy: string (choose from: 'clip', 'drop', 'interpolate', 'ffill', 'bfill', 'mean', 'median', 'smooth')

IMPORTANT:Return ONLY the JSON object below, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON:

{{
    "basic_stats": {{
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "trend": "string"
    }},
    "missing_info": {{
        "missing_count": int,
        "missing_percentage": float
    }},
    "outlier_info": {{
        "outlier_count": int,
        "outlier_percentage": float
    }},
    "quality_assessment": {{
        "data_quality_score": float,
        "main_issues": ["string"]
    }},
    "recommended_strategies": {{
        "missing_value_strategy": "string",
        "outlier_detect_strategy": "string",
        "outlier_handle_strategy": "string"
    }}
}}
"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            if not response.content or response.content.strip() == "":
                return self._generate_fallback_data_quality_analysis(data, outlier_info)
            
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            return self._generate_fallback_data_quality_analysis(data, outlier_info)
                
        except Exception as e:
            logger.error(f"Error in LLM data quality analysis: {e}")
            return self._generate_fallback_data_quality_analysis(data, outlier_info)
    
    def _generate_fallback_data_quality_analysis(self, data: pd.DataFrame, outlier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback data quality analysis when LLM fails"""
        logger.info("Generating fallback data quality analysis...")
        
        basic_stats = {
            "mean": float(data.mean().iloc[0]) if not data.empty else 0.0,
            "std": float(data.std().iloc[0]) if not data.empty else 0.0,
            "min": float(data.min().iloc[0]) if not data.empty else 0.0,
            "max": float(data.max().iloc[0]) if not data.empty else 0.0,
            "trend": "stable"
        }
        
        if len(data) > 1:
            try:
                slope = np.polyfit(range(len(data)), data.iloc[:, 0], 1)[0]
                if slope > 0.01:
                    basic_stats["trend"] = "increasing"
                elif slope < -0.01:
                    basic_stats["trend"] = "decreasing"
            except:
                pass
        
        missing_count = data.isnull().sum().sum()
        missing_percentage = (missing_count / (len(data) * len(data.columns))) * 100
        
        outlier_count = sum(info.get('count', 0) for info in outlier_info.values())
        outlier_percentage = (outlier_count / (len(data) * len(data.columns))) * 100
        
        quality_score = 1.0
        if missing_percentage > 0:
            quality_score -= missing_percentage / 100 * 0.3
        if outlier_percentage > 0.1:
            quality_score -= outlier_percentage * 0.2
        quality_score = max(0.0, min(1.0, quality_score))
        
        main_issues = []
        if missing_percentage > 0:
            main_issues.append("missing_values")
        if outlier_percentage > 0.05:
            main_issues.append("outliers")
        if len(main_issues) == 0:
            main_issues.append("none")
        
        return {
            "basic_stats": basic_stats,
            "missing_info": {
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_percentage)
            },
            "outlier_info": {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float(outlier_percentage)
            },
            "quality_assessment": {
                "data_quality_score": float(quality_score),
                "main_issues": main_issues
            },
            "recommended_strategies": {
                "missing_value_strategy": "interpolate" if missing_percentage > 0 else "none",
                "outlier_detect_strategy": "iqr",
                "outlier_handle_strategy": "clip" if outlier_percentage > 0 else "none"
            }
        }
    
    def _generate_visualizations(self, data: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Generate visualizations"""
        logger.info("Generating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        try:
            # Generate standard visualizations
            viz_configs = [
                {"name": "time_series", "type": "time_series", "title": "Time Series Plot"},
                {"name": "distribution", "type": "distribution", "title": "Distribution Analysis"},
                {"name": "rolling_stats", "type": "rolling_stats", "title": "Rolling Statistics", "window": 12},
                {"name": "autocorrelation", "type": "autocorrelation", "title": "Autocorrelation Analysis"},
                {"name": "seasonal_decomposition", "type": "seasonal_decomposition", "title": "Seasonal Decomposition", "period": 12}
            ]
            
            for viz_config in viz_configs:
                viz_name = viz_config["name"]
                viz_type = viz_config["type"]
                
                if viz_type == "time_series":
                    fig = self.visualizer.create_time_series_plot(data, viz_config["title"])
                elif viz_type == "distribution":
                    fig = self.visualizer.create_distribution_plot(data, viz_config["title"])
                elif viz_type == "rolling_stats":
                    fig = self.visualizer.create_rolling_stats_plot(data, viz_config.get("window", 12), viz_config["title"])
                elif viz_type == "autocorrelation":
                    fig = self.visualizer.create_autocorrelation_plot(data, viz_config["title"])
                elif viz_type == "seasonal_decomposition":
                    fig = self.visualizer.create_seasonal_decomposition_plot(data, viz_config.get("period", 12), viz_config["title"])
                else:
                    continue
                
                plot_path = output_path / f"{viz_name}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations[viz_name] = str(plot_path)
            
            logger.info(f"Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            visualizations = {}
        
        return visualizations
    
    def _save_preprocess_results(self, data: pd.DataFrame, analysis_report: Dict[str, Any], output_dir: str):
        """Save preprocessing results"""
        logger.info("Saving preprocessing results...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        data_path = output_path / "cleaned_data.csv"
        FileSaver.save_dataframe(data, data_path)
        logger.info(f"Cleaned data saved to {data_path}")
        
        # Save analysis report
        report_path = output_path / "analysis_report.json"
        FileSaver.save_json(analysis_report, report_path)
        logger.info(f"Analysis report saved to {report_path}")
        
        # Save preprocess summary
        summary = {
            'data_shape': data.shape,
            'data_columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'basic_stats': {
                'mean': data.mean().to_dict(),
                'std': data.std().to_dict(),
                'min': data.min().to_dict(),
                'max': data.max().to_dict(),
                'median': data.median().to_dict()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_path / "preprocess_summary.json"
        FileSaver.save_json(summary, summary_path)
        logger.info(f"Preprocess summary saved to {summary_path}")
    
    def _update_memory(self, data: pd.DataFrame, analysis_report: Dict[str, Any], visualizations: Dict[str, str]):
        """Update memory"""
        self.memory.store('cleaned_data', data, 'data')
        self.memory.store('analysis_report', analysis_report, 'analysis')
        self.memory.store('preprocess_visualizations', visualizations, 'visualizations')
        self.memory.store('preprocess_config', self.preprocess_config, 'config')
        
        # Record preprocessing history
        self.memory.add_history(
            'preprocess',
            {
                'data_shape': data.shape,
                'quality_score': analysis_report.get('quality_assessment', {}).get('data_quality_score', 0),
                'visualization_count': len(visualizations)
            }
        )
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get preprocessing summary"""
        analysis_report = self.memory.retrieve('analysis_report', 'analysis')
        if not analysis_report:
            return {}
        
        return {
            'data_shape': self.memory.retrieve('cleaned_data', 'data').shape if self.memory.retrieve('cleaned_data', 'data') is not None else None,
            'quality_score': analysis_report.get('quality_assessment', {}).get('data_quality_score', 0),
            'data_characteristics': analysis_report.get('quality_assessment', {}).get('data_characteristics', ''),
            'forecasting_readiness': analysis_report.get('forecasting_readiness', {}).get('data_suitability', ''),
            'preprocess_config': self.memory.retrieve('preprocess_config', 'config')
        }