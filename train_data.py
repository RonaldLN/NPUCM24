# 使用tensorflow搭建一个MLP模型，用于训练数据

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# 读取数据
def read_data():
    names = ['附件1：Normal_exp', '附件2：EarlyStage_exp', '附件3：LaterStage_exp']
    data = []
    for n in names:
        classfied = int(n[2]) - 1
        d = np.load(n + '_t.npy', allow_pickle=True)[1:, :]  # 去掉第一行(基因编号)
        # 添加标签
        d = np.insert(d, d.shape[1], classfied, axis=1)
        # print(d)
        data.append(d)
    return data

# 将3个array数组合并
def merge_data(data):
    data = np.concatenate(data, axis=0)
    # print(data)
    # print(data.shape)
    np.save('merged_data.npy', data)
    return data

# 数据预处理
def preprocess(data):
    # 随机打乱数据
    data = shuffle(data, random_state=42)
    print(data)

    # 将数据分为特征和标签
    X = data[:, :-1]
    y = data[:, -1]

    print(X, y)

    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    return X_train, X_test, y_train, y_test

"""
说明：
标签为0，1，2，分别代表Normal_exp，EarlyStage_exp，LaterStage_exp
"""

# 构建MLP模型
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# 训练模型
def train_model(X_train, y_train, epochs):
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy: ', acc)

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ')
    print(cm)

    cr = classification_report(y_test, y_pred)
    print('Classification Report: ')
    print(cr)

    # y_pred = model.predict(X_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_test = y_test.astype('int32')
    # print(y_test)
    # print(y_pred)
    # # roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
    # # print('ROC AUC: ', roc_auc)

    # # fpr = dict()
    # # tpr = dict()
    # # roc_auc = dict()
    # # for i in range(3):
    # #     fpr[i], tpr[i], _ = roc_curve(y_test, y_pred, pos_label=i)
    # #     roc_auc[i] = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
    # # print(fpr)
    # # print(tpr)
    # # print(roc_auc)

    # plt.figure()
    # lw = 2
    # # colors = ['aqua', 'darkorange', 'cornflowerblue']
    # # for i, color in zip(range(3), colors):
    # #     plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()

    return acc


if __name__ == '__main__':
    # data = read_data()
    # data = merge_data(data)
    data = np.load('merged_data.npy', allow_pickle=True)
    X_train, X_test, y_train, y_test = preprocess(data)
    epochs = 400
    model = train_model(X_train, y_train, epochs)
    acc = evaluate_model(model, X_test, y_test)
    print('Accuracy: ', acc)
    model.save(f'model_{epochs}.h5')
