import xgboost
import numpy as np
import shap
import pickle
import pandas as pd

from train_data import preprocess

# convert `X_train` to pd.DataFrame, and add headers
def add_headers(X_train):
    headers = np.load('headers.npy', allow_pickle=True)
    X_train = pd.DataFrame(X_train, columns=headers)
    return X_train

# get a dataset on income prediction
# X, y = shap.datasets.adult()
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X_train = add_headers(X_train)

X = X_train
y = y_train

# train an XGBoost model (but any other model type would also work)
# model = xgboost.XGBClassifier()
# model.fit(X, y)

# with open('xgboost_model.bin', 'wb') as f:
#     pickle.dump(model, f)

model = xgboost.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
model.load_model("xgboost_model.json")

X = X_train[:5]
y = y_train[:5]

# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, X)
shap_values = explainer(X[:100])

with open('shap_values_xgboost_exact.bin', 'wb') as f:
    pickle.dump(shap_values, f)

# get just the explanations for the positive class
shap_values = shap_values[..., 1]

shap.plots.beeswarm(shap_values)

shap.plots.heatmap(shap_values)

shap.plots.bar(shap_values)

shap.plots.waterfall(shap_values[0])

# ValueError: It takes 2.87*10^3963 masked evaluations to run the Exact explainer on this instance, but max_evals=100000!
