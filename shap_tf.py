import shap
import numpy as np
from train_data import preprocess, build_model

# read and preprocess data
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, _, _ = preprocess(data)

# select a set of background examples to take an expectation over
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# build model
model = build_model(X_train.shape[1])
model.load_weights('model_400.h5')

# Assuming 'model' is your deep learning model and 'background' is a sample of your input data
explainer = shap.DeepExplainer(model, background)

# 'test_data' is the data you want to explain
shap_values = explainer.shap_values(X_test)

# plot the feature attributions
shap.image_plot(shap_values, -X_test[:])
