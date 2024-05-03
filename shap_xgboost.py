import xgboost
import numpy as np
import shap
import pickle

from train_data import preprocess

# get a dataset on income prediction
# X, y = shap.datasets.adult()
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X = X_train
y = y_train

# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier()
model.fit(X, y)

with open('xgboost_model.bin', 'wb') as f:
    pickle.dump(model, f)

X = X_train[:5]
y = y_train[:5]

# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(model.predict_proba, X, max_evals=30000)
shap_values = explainer(X[:100])

with open('shap_values_xgboost.bin', 'wb') as f:
    pickle.dump(shap_values, f)

# get just the explanations for the positive class
shap_values = shap_values[..., 1]

shap.plots.bar(shap_values)

shap.plots.waterfall(shap_values[0])