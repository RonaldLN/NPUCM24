# 使用PyTorch搭建一个MLP模型，用于训练数据

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# # 读取数据
# def read_data():
#     # ... 代码保持不变 ...

# # 将3个array数组合并
# def merge_data(data):
#     # ... 代码保持不变 ...

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

    # 将数据转换为PyTorch张量
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train.astype(np.int64))
    y_test = torch.from_numpy(y_test.astype(np.int64))

    return X_train, X_test, y_train, y_test

# 构建MLP模型
class MLP(nn.Module):
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# 训练模型
def train_model(X_train, y_train, epochs):
    model = MLP(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # print information
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, y_pred = torch.max(outputs, 1)

    # ... 代码保持不变 ...
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy: ', acc)

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ')
    print(cm)

    cr = classification_report(y_test, y_pred)
    print('Classification Report: ')
    print(cr)

    return acc

if __name__ == '__main__':
    # data = read_data()
    # data = merge_data(data)
    data = np.load('merged_data.npy', allow_pickle=True)
    X_train, X_test, y_train, y_test = preprocess(data)
    epochs = 200
    model = train_model(X_train, y_train, epochs)
    acc = evaluate_model(model, X_test, y_test)
    print('Accuracy: ', acc)
    torch.save(model.state_dict(), f'model_{epochs}.pt')