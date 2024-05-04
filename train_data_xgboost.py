import xgboost
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from train_data import preprocess

# get a dataset on income prediction
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X = X_train
y = y_train

# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
# model.fit(X, y)
model.load_model("xgboost_model.json")

preds =  model.predict(X_test)

print(accuracy_score(y_test, preds))

cr = classification_report(y_test, preds)
print(cr)

### Compute ROC curve and ROC area for each class
### -------------------
fpr = dict()
tpr = dict()
# roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test, preds, pos_label=i)
    # roc_auc[i] = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
print(fpr)
print(tpr)
# print(roc_auc)

plt.figure()
lw = 2
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0}'.format(i))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()
### -------------------

# model.save_model("xgboost_model.json")