import numpy as np
import shap
import pickle

with open('shap_values_xgboost.bin', 'rb') as f:
    shap_values = pickle.load(f)

shap_values = shap_values[..., 1]

shap.plots.bar(shap_values)

shap.plots.waterfall(shap_values[0])