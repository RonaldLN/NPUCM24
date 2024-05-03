import xgboost
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from train_data import preprocess

# get a dataset on income prediction
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X = X_train
y = y_train

# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
model.fit(X, y)

preds =  model.predict(X_test)

print(accuracy_score(y_test, preds))

cr = classification_report(y_test, preds)
print(cr)

model.save_model("xgboost_model.json")