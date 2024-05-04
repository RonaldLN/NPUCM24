import shap
import numpy as np
from train_data import preprocess, build_model

# read and preprocess data
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, _, _ = preprocess(data)

# build model
model = build_model(X_train.shape[1])
model.load_weights('model_400.h5')

X = X_test[:]

def f(X):
    return model.predict(X).flatten()

explainer = shap.KernelExplainer(model, X)
shap_values = explainer.shap_values(X, nsamples=15000)
shap.force_plot(explainer.expected_value, shap_values, X)
