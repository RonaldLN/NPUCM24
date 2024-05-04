import xgboost
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from train_data import preprocess

high_genes_indices = [707, 5626, 6264, 4510, 11544, 10703, 1764, 10875, 6204]
high_genes = ['SPRY2', 'PPP1R12B', 'SLC30A7', 'HOXA7', 'SCN4B', 'KANK1', 'SDPR', 'TMEM220', 'ATP5F1']

# get a dataset on income prediction
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X = X_train
y = y_train

for g, idx in zip(high_genes, high_genes_indices):
    # only use the gene with index idx
    X_i = X[:, idx:idx + 1]
    X_test_i = X_test[:, idx:idx + 1]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X_i, y)

    preds =  model.predict(X_test_i)

    print(f"xgboost_model_{idx}_{g}:")
    print(accuracy_score(y_test, preds))

    cr = classification_report(y_test, preds)
    print(cr)

    # ### Compute ROC curve and ROC area for each class
    # ### -------------------
    # fpr = dict()
    # tpr = dict()
    # # roc_auc = dict()
    # for i in range(3):
    #     fpr[i], tpr[i], _ = roc_curve(y_test, preds, pos_label=i)
    #     # roc_auc[i] = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    # print(fpr)
    # print(tpr)
    # # print(roc_auc)

    # plt.figure()
    # lw = 2
    # colors = ['aqua', 'darkorange', 'cornflowerblue']
    # for i, color in zip(range(3), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0}'.format(i))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.savefig('roc_curve.png')
    # plt.show()
    # ### -------------------

    model.save_model(f"xgboost_model_{idx}_{g}.json")