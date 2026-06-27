# GraphTransformer XAI Analysis Report

**Generated on:** 2025-12-08 14:33:34

## Model Information
- Model type: GraphTransformer
- Number of features: 645
## XAI Analysis Results

### SHAP
- Status: Failed - Unable to allocate 1.34 GiB for an array with shape (7740, 23220) and data type float64

### LIME
- num_explanations: 3
- samples_explained: 0, 1, 2

### CAUSAL
- Status: Completed successfully

### FEATURE_IMPORTANCE
- Status: Failed - could not broadcast input array from shape (23220,) into shape (645,)

### ATTENTION
- Status: Completed successfully

### DICE
- Status: Failed - only integer scalar arrays can be converted to a scalar index

## Key Insights
- Graph structure analyzed with 645 nodes
- Feature importance analysis reveals inter-feature dependencies
- Counterfactual analysis shows sensitivity to input perturbations
- LIME provides interpretable local explanations

## Generated Files
- SHAP: `model/GraphTransformer/xai/shap/`
- LIME: `model/GraphTransformer/xai/lime/`
- CAUSAL: `model/GraphTransformer/xai/causal/`
- FEATURE_IMPORTANCE: `model/GraphTransformer/xai/feature_importance/`
- ATTENTION: `model/GraphTransformer/xai/attention/`
- DICE: `model/GraphTransformer/xai/dice/`
- REPORT: `model/GraphTransformer/xai/report/`
