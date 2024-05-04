import numpy as np
import shap
import pickle

with open('shap_values_xgboost_permutation.bin', 'rb') as f:
    shap_values = pickle.load(f)

shap_values = shap_values[..., 1]

# # shap.plots.beeswarm(shap_values)

# shap.plots.heatmap(shap_values, max_display=10)

# shap.plots.bar(shap_values, max_display=10)

# shap.plots.waterfall(shap_values[0], max_display=10)

gene_names = shap_values.feature_names
high_genes = ['SPRY2', 'PPP1R12B', 'SLC30A7', 'HOXA7', 'SCN4B', 'KANK1', 'SDPR', 'TMEM220', 'ATP5F1']
high_genes_indices = [gene_names.index(gene) for gene in high_genes]

for g, idx in zip(high_genes, high_genes_indices):
    print(f'{idx}: {g}')

print(high_genes_indices)