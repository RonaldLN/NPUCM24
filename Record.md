---
draft: true
date: 2024-05-01
authors:
  - ronald_luo
categories:
  - Chronicle of Events
comments: true
---

# 24数模校赛记录

(A题)题目分别给出了正常人、早期和晚期癌症患者的基因表达数据(一个数值)，然后大概需要分析出能够辨别癌症状态的特征基因，

而我觉得这题能可以使用神经网络来操作，所以就打算选这题尝试

>   另一方面也因为，B题是*基于负面影响最小的任务分配问题*，感觉没有太多可以操作的空间

<!-- more -->

## 寻找分析神经网络模型特征的方法

我的想法是使用MLP来训练数据，但由于这题最后需要找出特征，而之前不太清楚要如何由已训练好的模型找到特征，所以搜寻信息

???+ example "对话记录"

    你:
    
    How to extract feature dimensions from DNN models?
    
    &nbsp;
    
    Bing:
    
    要通过深度学习来确定数据中哪些维度是重要的，可以采用以下方法：
    
    1.   [**特征重要性评估**：使用类似于XGBoost模型中的特征重要性计算方法。在深度神经网络（DNN）中，可以通过观察模型中每个特征的变化对模型性能的影响来评估特征的重要性。例如，可以通过随机打乱每个特征的值并重新评估模型性能来实现。如果性能下降显著，则该特征可能很重要](https://www.zhihu.com/question/310837513) [`¹`](https://www.zhihu.com/question/310837513) [`²`](https://blog.csdn.net/fengdu78/article/details/120735471)。
    2.   [**Shapley值**：Shapley值是一种博弈论中的概念，用于量化每个玩家（在这里是特征）对游戏（模型的预测）的贡献。在深度学习中，可以使用Shapley值来估计每个特征对模型预测的贡献](https://www.zhihu.com/question/310837513) [`¹`](https://www.zhihu.com/question/310837513)。
    3.   **正则化技术**：使用L1正则化可以帮助进行特征选择，因为它倾向于产生稀疏权重矩阵。在训练过程中，不重要的特征的权重会趋近于零。
    4.   **注意力机制**：在深度学习模型中，特别是在序列模型如RNN中，注意力机制可以用来识别对预测结果最有影响的输入部分。
    5.   **模型可解释性工具**：使用模型可解释性工具，如LIME或SHAP，可以帮助解释模型的预测，并指出哪些特征对预测结果有最大影响。
    
    通过这些方法，你可以识别出对模型预测最有影响的特征，即数据中的重要维度。这些方法可以单独使用，也可以结合使用，以获得更全面的特征重要性评估。
    
    了解详细信息 [`1 | zhihu.com`](https://www.zhihu.com/question/310837513) [`2 | blog.csdn.net`](https://blog.csdn.net/fengdu78/article/details/120735471) [`3 | zhuanlan.zhihu.com`](https://zhuanlan.zhihu.com/p/395560780) [`4 | bing.com`](https://bing.com/search?q=深度学习+如何获取数据重要维度) [`5 | zhuanlan.zhihu.com`](https://zhuanlan.zhihu.com/p/655633788)

查看了Bing提到的知乎上一个问题的回答，注意到这个回答提到了**==Shapley值==**

[使用DNN训练神经网络模型时，如何知道每个特征的重要性（像xgboost模型能计算出特征重要性一样）？ - 王睿的回答 - 知乎](https://www.zhihu.com/question/310837513/answer/1281609735)

而然后重新使用英文询问Bing，**也提到了Shapley值**

???+ example "对话记录"

    你:
    
    How to extract feature dimensions from DNN models?
    
    &nbsp;
    
    Bing:
    
    To extract feature dimensions from Deep Neural Network (DNN) models, you typically follow these steps:
    
    1.   **Choose the layer** from which you want to extract features. This is often a layer towards the end of the network, just before the fully connected layers, as it captures high-level features.
    2.   **Pass your input data** through the network up to the chosen layer. You can do this by modifying the network architecture to output the desired layer’s activations.
    3.   **Collect the output** of that layer, which will be the features. The shape of this output will give you the feature dimensions.
    
    For example, in Keras, you can use the `Model` class to create a new model that outputs the features from a specific layer:
    
    ```python
    from keras.models import Model
    
    # Assume 'base_model' is your pre-trained DNN model
    # and 'layer_name' is the name of the layer from which you want to extract features.
    
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    
    # Now you can use this model to predict and extract features
    features = model.predict(your_data)
    ```
    
    The `features` variable will now contain the output of the layer specified, and the shape of this array will give you the feature dimensions.
    
    For large datasets that don’t fit into memory, you can use **incremental learning**. [This involves extracting features in batches and possibly storing them on disk before training a classifier on top of these features](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/) [`¹`](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/).
    
    [If you’re working with Convolutional Neural Networks (CNNs), the filters in the convolutional layers act as feature extractors, capturing patterns such as edges and textures in the initial layers, and more complex features in deeper layers](https://medium.com/@leicao.me/how-deep-learning-neural-networks-extracts-features-277244bcf66e) [`²`](https://medium.com/@leicao.me/how-deep-learning-neural-networks-extracts-features-277244bcf66e).
    
    Remember, the exact method may vary depending on the framework you’re using and the specific architecture of your DNN model. The provided example is a general approach and might need adjustments based on your setup.
    
    了解详细信息 [`1 | pyimagesearch.com`](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/) [`2 | medium.com`](https://medium.com/@leicao.me/how-deep-learning-neural-networks-extracts-features-277244bcf66e) [`3 | stats.stackexchange.com`](https://stats.stackexchange.com/questions/473667/feature-extraction-in-deep-neural-networks) [`4 | stackoverflow.com`](https://stackoverflow.com/questions/69298074/how-to-extract-features-from-a-cnn-model-using-its-actuall-weights-after-trainin) [`5 | nature.com`](https://www.nature.com/articles/s41598-018-34833-6.pdf)
    
    &nbsp;
    
    你:
    
    For example, I have a dataset with 10 dimensions each, which can be used to classify the data into one of three categories. How can I determine which dimension has the greatest impact on classification (using deep neural networks)
    
    &nbsp;
    
    Bing:
    
    To determine which dimension has the greatest impact on classification using deep neural networks, you can use techniques such as **permutation importance** or **SHAP (SHapley Additive exPlanations)** values. Here’s a brief overview of each method:
    
    <big>**Permutation Importance**</big>
    
    Permutation importance is a model inspection technique that can be used for any fitted estimator when the data is tabular. This method involves shuffling individual feature columns and observing the impact on the model’s accuracy. [The idea is that the features that are important to the model’s predictions will show a significant decrease in model performance when they are shuffled](https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python) [`¹`](https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python).
    
    Here’s how you can implement permutation importance using Python with Keras:
    
    ```python
    from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
    import eli5
    from eli5.sklearn import PermutationImportance
    
    # Define your model
    def base_model():
        model = Sequential()
        model.add(Dense(200, input_dim=10, activation='relu'))
        model.add(Dense(3, activation='softmax'))  # Assuming 3 categories
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    # Fit your model
    X = ...  # Your input features
    y = ...  # Your target labels
    my_model = KerasRegressor(build_fn=base_model, epochs=100, batch_size=5, verbose=0)
    my_model.fit(X, y)
    
    # Calculate permutation importance
    perm = PermutationImportance(my_model, random_state=1).fit(X, y)
    eli5.show_weights(perm, feature_names=['Feature 1', 'Feature 2', ..., 'Feature 10'])
    ```
    
    <big>**SHAP Values**</big>
    
    SHAP values provide a way to understand the contribution of each feature to the model’s predictions. [It is based on game theory and calculates the contribution of each feature to the prediction of each instance in comparison to the average prediction for the dataset](https://aitechtrend.com/a-guide-to-explaining-feature-importance-in-neural-networks-using-shap/) [`²`](https://aitechtrend.com/a-guide-to-explaining-feature-importance-in-neural-networks-using-shap/).
    
    Here’s a simple example of how to use SHAP with a neural network:
    
    ```python
    import shap
    
    # Assume 'model' is your trained Keras model and 'X' is your input data
    
    # Explain predictions using SHAP
    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # Plot the SHAP values for the first instance
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0])
    ```
    
    Both methods can give you insights into which dimensions (features) are most important for your deep neural network’s classification decisions. Remember to carefully interpret the results, as the importance of features can depend on the model and the specific data you are working with.
    
    了解详细信息 [`1 | stackoverflow.com`](https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python) [`2 | aitechtrend.com`](https://aitechtrend.com/a-guide-to-explaining-feature-importance-in-neural-networks-using-shap/) [`3 | machinelearningmastery.com`](https://machinelearningmastery.com/calculate-feature-importance-with-python/) [`4 | datascience.stackexchange.com`](https://datascience.stackexchange.com/questions/44644/how-to-determine-feature-importance-in-a-neural-network) [`5 | stats.stackexchange.com`](https://stats.stackexchange.com/questions/261008/deep-learning-how-do-i-know-which-variables-are-important)

于是打算尝试这个方法，

但由于Bing给出的代码中，应该是需要训练好模型之后再对模型进行分析，于是就开始编写python代码处理数据并训练模型

## 处理数据和训练

copilot在数据预处理部分中给了这样的代码

```python
# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

不太清楚*标准化数据*的含义，于是让copilot对此进行解释

???+ example "对话记录"

    RonaldLN:
    
    `@workspace` `/explain` Write an explanation for the active selection as paragraphs of text.
    
    ```python
    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ```
    
    &nbsp;
    
    Github Copilot:
    
    ??? quote "已使用 3 引用"
    
        -   train_data.py: 48-50
        -   .../sklearn/base.py: 1043
        -   .../sklearn/preprocessing/_base.py: 694
    
    这段代码的目的是对数据进行标准化处理。标准化是一种常见的数据预处理方法，它可以使得不同规模和量纲的数据转化为相同的规模，便于进行后续的数据分析或者模型训练。
    
    首先，`StandardScaler()`是来自`sklearn.preprocessing`模块的一个类，用于实现数据的标准化处理。这个类的实例化结果被赋值给了变量`scaler`。
    
    然后，`scaler.fit_transform(X)`这个函数调用做了两件事情。首先，`fit`方法会计算数据`X`的均值和标准差。然后，`transform`方法会使用这些计算出的均值和标准差来对数据`X`进行标准化处理。标准化处理的结果会替换原来的`X`。
    
    标准化处理后的数据`X`的每个特征（列）的均值为0，标准差为1。这样处理后的数据更适合作为机器学习算法的输入，因为许多机器学习算法都假设输入数据的特征是均值为0，标准差为1的。

---

在评估模型部分中，copilot生成了这样的代码

```python
cr = classification_report(y_test, y_pred)
print('Classification Report: ')
print(cr)
```

输出的显示是这样的

```bash
Classification Report: 
              precision    recall  f1-score   support

         0.0       0.96      1.00      0.98        23
         1.0       0.41      0.44      0.42        25
         2.0       0.55      0.50      0.52        32

    accuracy                           0.62        80
   macro avg       0.64      0.65      0.64        80
weighted avg       0.62      0.62      0.62        80
```

不太清楚具体的含义，于是询问copilot

???+ example "对话记录"

    RonaldLN:
    
    ```bash
    Classification Report: 
                  precision    recall  f1-score   support
    
             0.0       0.96      1.00      0.98        23
             1.0       0.41      0.44      0.42        25
             2.0       0.55      0.50      0.52        32
    
        accuracy                           0.62        80
       macro avg       0.64      0.65      0.64        80
    weighted avg       0.62      0.62      0.62        80
    ```
    
    解释一下这个
    
    &nbsp;
    
    Github Copilot:
    
    ??? quote "已使用 1 引用"
    
        -   train_data.py: 85-104
    
    这是一个分类报告，它提供了模型在每个类别上的性能度量。
    
    -   `precision`：精确率，表示被模型预测为正类的样本中，实际上也是正类的比例。例如，对于类别0，精确率为0.96，表示模型预测为类别0的样本中，96%实际上是类别0。
    -   `recall`：召回率，表示实际上是正类的样本中，被模型正确预测的比例。例如，对于类别0，召回率为1.00，表示所有实际上是类别0的样本，都被模型正确预测了。
    -   `f1-score`：F1分数，是精确率和召回率的调和平均值，用于同时考虑精确率和召回率的性能度量。例如，对于类别0，F1分数为0.98，表示精确率和召回率的调和平均值为0.98。
    -   `support`：支持度，表示每个类别的样本数量。例如，类别0有23个样本，类别1有25个样本，类别2有32个样本。
    -   `accuracy`：准确率，表示所有样本中，被模型正确预测的比例。在这个例子中，准确率为0.62，表示62%的样本被正确预测。
    -   `macro avg`：宏平均，表示所有类别的平均精确率、召回率和F1分数，每个类别的权重相同。
    -   `weighted avg`：加权平均，表示所有类别的平均精确率、召回率和F1分数，每个类别的权重与其样本数量成比例。
    
    从这个报告中，我们可以看出，模型在类别0上的性能最好，精确率、召回率和F1分数都很高。然而，对于类别1和2，模型的性能较差。

---

进行训练之后发现，`epochs` 设置成200或300时预测的效果比较好

---

>   由于不清楚为什么输出层有3个神经元，于是询问了一下copilot

???+ example "对话记录"

    RonaldLN:
    
    `@workspace` `/explain` Write an explanation for the active selection as paragraphs of text.
    
    ```python
    # 构建MLP模型
    def build_model(input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
    ```
    
    &nbsp;
    
    GitHub Copilot: 
    
    这段代码定义了一个函数`build_model`，用于构建一个多层感知器（MLP）模型。MLP是一种前馈神经网络，它包含至少三层（一个输入层，一个或多个隐藏层，一个输出层）的节点。每一层都完全连接到下一层。
    
    在这个函数中，我们首先创建一个`Sequential`模型。`Sequential`模型是Keras中最简单的模型，它由多个网络层线性堆叠。
    
    然后，我们添加了四个`Dense`层。`Dense`层是最常用的层，它是全连接层，即这一层的每个神经元都与上一层的所有神经元相连。
    
    -   第一层有128个神经元，激活函数为'relu'，输入形状为`input_shape`。'relu'是一种常用的激活函数，它可以增加模型的非线性。输入形状`input_shape`是一个元组，表示输入数据的维度。
    -   第二层有64个神经元，激活函数为'relu'。
    -   第三层有32个神经元，激活函数为'relu'。
    -   最后一层是输出层，有3个神经元，激活函数为'softmax'。'softmax'函数可以将一个实数向量转换为概率分布，因此这个模型的输出可以被解释为每个类别的概率。
    
    这个函数最后返回构建好的模型。
    
    &nbsp;
    
    RonaldLN:
    
    输出层为什么有3个神经元？
    
    &nbsp;
    
    GitHub Copilot:
    
    输出层的神经元数量通常与你要解决的问题的类别数量相对应。在这个例子中，输出层有3个神经元，这意味着你的模型是在解决一个三分类问题。
    
    在多分类问题中，我们通常使用softmax激活函数，它可以将模型的输出转换为概率分布，这样每个神经元的输出就可以被解释为对应类别的概率。在这个例子中，模型的输出是一个包含三个元素的向量，每个元素代表一个类别的概率。

## 尝试使用SHAP值对模型进行分析

让chatgpt根据 `train_data.py` 写使用shap分析模型的python代码

???+ example "对话记录"

    You:
    
    我想使用SHAP值来分析 `train_data.py` 中训练出来的模型的关键的特征，你能帮我编写一个新的文件吗？
    
    ```python title="train_data.py"
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
    ```
    
    &nbsp;
    
    ChatGPT:
    
    好的，你想要一个新的文件来使用 SHAP（SHapley Additive exPlanations）来解释模型的关键特征。SHAP 值可以帮助你理解模型对于每个特征的预测贡献。下面是一个示例文件 `shap_analysis.py`，用于解释 `train_data.py` 中训练出来的模型：
    
    ```python
    import numpy as np
    import shap
    from train_data import preprocess, build_model
    
    # 读取并预处理数据
    data = np.load('merged_data.npy', allow_pickle=True)
    X_train, _, _, _ = preprocess(data)
    
    # 构建模型
    model = build_model(X_train.shape[1])
    model.load_weights('model_400.h5')  # 加载训练好的模型权重
    
    # 初始化一个 SHAP 解释器
    explainer = shap.Explainer(model, X_train)
    
    # 计算 SHAP 值
    shap_values = explainer(X_train)
    
    # 可视化 SHAP 值
    shap.summary_plot(shap_values, X_train, feature_names=[f'Feature {i}' for i in range(X_train.shape[1])])
    ```
    
    这个文件首先加载了训练好的模型和数据，然后使用 SHAP 解释器来计算并可视化 SHAP 值。你可以运行这个文件来查看每个特征对于模型预测的影响程度。确保安装了 `shap` 库，你可以使用 `pip install shap` 来安装。
    
    &nbsp;
    
    You:
    
    ```bash
    Traceback (most recent call last):
      File "c:\PycharmProjects\NPUCM24\testshap.py", line 17, in <module>
        shap_values = explainer(X_train)
      File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_permutation.py", line 77, in __call__
        return super().__call__(
      File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_explainer.py", line 266, in __call__
        row_result = self.explain_row(
      File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_permutation.py", line 157, in explain_row
        raise ValueError(f"max_evals={max_evals} is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = {2 * len(inds) + 1}!")
    ValueError: max_evals=500 is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = 26327!
    ```
    
    显示了上面这样的报错，应该如何修改？
    
    &nbsp;
    
    ChatGPT:
    
    这个报错提示 `max_evals` 参数设置过低，`max_evals` 是 Permutation Explainer 中的一个参数，用于控制计算 SHAP 值时的采样次数。根据报错信息，你的数据集具有 13163 个特征，所以至少需要 `2 * num_features + 1 = 26327` 次采样。
    
    你可以在初始化解释器时传入一个更大的 `max_evals` 参数来解决这个问题。例如：
    
    ```python
    # 初始化一个 SHAP 解释器，并指定更大的 max_evals 参数
    explainer = shap.Explainer(model, X_train, algorithm="permutation", max_evals=50000)
    ```
    
    通过增加 `max_evals` 参数的值，你可以确保 SHAP 解释器能够充分地对模型进行解释。

然后再次进行运行时，显示了

```bash
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 258. GiB for an array with shape (2632700, 13163) and data type float64
```

猜测是内存不够的原因，

于是添加了一行

```python
X_train = X_train[:3]
```

然后就可以运行了，但是显示了另外一个报错

```bash
Traceback (most recent call last):
  File "c:\PycharmProjects\NPUCM24\testshap.py", line 23, in <module>
    shap.summary_plot(shap_values, X_train, feature_names=[f'Feature {i}' for i in range(X_train.shape[1])])
  File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\plots\_beeswarm.py", line 607, in summary_legacy
    feature_names=feature_names[sort_inds],
TypeError: only integer scalar arrays can be converted to a scalar index
```

然后由于每次运行，*计算 SHAP 值* 都需要等几分钟(不方便测试 `summary_plot` )，所以想把 `shap_values` 保存成二进制文件，就可以直接加载不需要再次运算(但不清楚具体操作方法)，所以询问chatgpt

???+ example "对话记录"

    You:
    
    i want to save a variable to a binary file in python, how should i do?
    
    &nbsp;
    
    ChatGPT:
    
    You can save a variable to a binary file in Python using the `pickle` module. Here's a simple example:
    
    ```python
    import pickle
    
    # Define your variable
    data = {'name': 'John', 'age': 30, 'city': 'New York'}
    
    # Open a file in binary mode
    with open('data.bin', 'wb') as f:
        # Dump the variable to the file
        pickle.dump(data, f)
    ```
    
    This code saves the `data` variable to a binary file named `'data.bin'`. You can then load this variable back from the file using `pickle.load()` function. Remember to always open the file in binary mode (`'rb'` for reading and `'wb'` for writing) when working with binary files.

于是仿照chatgpt提供的代码，将 `shap_values` 保存成了 `shap_values.bin`

```python
# 将 `shap_values` 保存为二进制文件
with open('shap_values.bin', 'wb') as f:
    pickle.dump(shap_values, f)
```

>   读取这个二进制文件的方法为
>
>   ```python
>   with open('shap_values.bin', 'rb') as f:
>       shap_values = pickle.load(f)
>   ```

---

>   其间发现 `shap_values` 的类型是 `shap._explanation.Explanation`

---

由于一直尝试无果后，去查看[官方仓库 README.md](https://github.com/shap/shap/)，

>   过程中发现，无论是github copilot 、 chatgpt 还是 bing，似乎对 `shap` 包的使用，在给出的回答中的代码都不是很正确，我个人猜测可能是由于这个包比较新的原因(？)，并且也感到反而bing给出的回答似乎更加正确一些，
>
>   所以最后还是决定自己去查看官方的说明

发现应该参照 [`Deep learning example with DeepExplainer (TensorFlow/Keras models)`](https://github.com/shap/shap/#deep-learning-example-with-deepexplainer-tensorflowkeras-models) 处的代码结构，并且应该使用的(解释器)是 `shap.DeepExplainer` ，于是

```python title="shap_tf.py" linenums="1"
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
```

但是在运行时，先是显示

```bash
Traceback (most recent call last):
  File "c:\PycharmProjects\NPUCM24\shap_tf.py", line 17, in <module>
    explainer = shap.DeepExplainer(model, background)
  File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_deep\__init__.py", line 84, in __init__
    self.explainer = TFDeep(model, data, session, learning_phase_flags)
  File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_deep\deep_tf.py", line 172, in __init__
    noutputs = self.model_output.shape.as_list()[1]
AttributeError: 'tuple' object has no attribute 'as_list'
```

尝试之后认为应该是 `shap` 包的问题，所以修改它的代码

```python title=".../_deep/deep_tf.py" linenums="172" hl_lines="173"
# noutputs = self.model_output.shape.as_list()[1]
noutputs = list(self.model_output.shape)[1]
```

再次运行，原报错消失，但出现新的报错

```bash
Traceback (most recent call last):
  File "c:\PycharmProjects\NPUCM24\shap_tf.py", line 20, in <module>
    shap_values = explainer.shap_values(X_test)
  ...
  File "C:\Users\Ronald\AppData\Local\Temp\__autograph_generated_file4ldlekdk.py", line 12, in tf__grad_graph
    phase = ag__.converted_call(ag__.ld(tf).keras.backend.learning_phase, (), None, fscope)
AttributeError: in user code:

    File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\explainers\_deep\deep_tf.py", line 240, in grad_graph  *
        phase = tf.keras.backend.learning_phase()

    AttributeError: module 'keras._tf_keras.keras.backend' has no attribute 'learning_phase'
```

好像又是包本身代码的问题，询问copilot如何解决

???+ example "对话记录"

    RonaldLN:
    
    ```bash
    AttributeError: module 'keras._tf_keras.keras.backend' has no attribute 'learning_phase'
    ```
    
    &nbsp;
    
    Github Copilot:
    
    这个错误是因为你试图访问`keras._tf_keras.keras.backend`模块中不存在的`learning_phase`属性。这可能是因为你使用的Keras版本与代码中使用的版本不兼容。
    
    在TensorFlow 2.x和新版本的Keras中，`learning_phase`已经被移除。在训练和预测模式下，模型的行为现在由内置的Keras方法（如`fit`、`evaluate`和`predict`）自动管理。
    
    如果你的代码依赖于`learning_phase`，你可能需要更新你的代码以适应新版本的Keras和TensorFlow。具体来说，你可能需要移除所有引用`learning_phase`的代码，并确保你的模型在训练和预测时使用正确的方法。
    
    如果你不能修改代码，另一个解决方案是降级你的Keras和TensorFlow版本到一个包含`learning_phase`的版本。你可以使用如下命令来安装特定版本的Keras和TensorFlow：
    
    ```bash
    pip install keras==<version> tensorflow==<version>
    ```
    
    你需要将`<version>`替换为你想要安装的版本号。

---

感觉这个问题不是很好解决，

于是就想试试换成用PyTorch来搭建MLP模型并训练，看看pytorch的模型能不能正常跑，

于是让copilot转换 `train_data.py`

???+ example "对话记录"

    SeazSu:
    
    你可以帮我将 `train_data.py` 中的代码改写成用pytorch来实现吗？
    
    &nbsp;
    
    GitHub Copilot:
    
    ??? quote "已使用 1 引用"
    
        -   train_data.py: 1-149
    
    当然可以，以下是使用PyTorch重写的代码：
    
    ```python
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
    
    # 读取数据
    def read_data():
        # ... 代码保持不变 ...
    
    # 将3个array数组合并
    def merge_data(data):
        # ... 代码保持不变 ...
    
    # 数据预处理
    def preprocess(data):
        # ... 代码保持不变 ...
    
        # 将数据转换为PyTorch张量
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
    
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
    
        return model
    
    # 评估模型
    def evaluate_model(model, X_test, y_test):
        with torch.no_grad():
            outputs = model(X_test)
            _, y_pred = torch.max(outputs, 1)
    
        # ... 代码保持不变 ...
    
    if __name__ == '__main__':
        # data = read_data()
        # data = merge_data(data)
        data = np.load('merged_data.npy', allow_pickle=True)
        X_train, X_test, y_train, y_test = preprocess(data)
        epochs = 400
        model = train_model(X_train, y_train, epochs)
        acc = evaluate_model(model, X_test, y_test)
        print('Accuracy: ', acc)
        torch.save(model.state_dict(), f'model_{epochs}.pt')
    ```
    
    这个代码使用了PyTorch的`nn.Module`类来定义MLP模型，并使用了`optim.Adam`优化器和`nn.CrossEntropyLoss`损失函数。训练数据被封装在一个`DataLoader`对象中，该对象在每个epoch中迭代数据并更新模型的权重。最后，模型的权重被保存为一个`.pt`文件。
    
    &nbsp;
    
    SeazSu:
    
    ```bash
    Traceback (most recent call last):
      File "c:\PycharmProjects\NPUCM24\train_data_torch.py", line 112, in <module>
        X_train, X_test, y_train, y_test = preprocess(data)
      File "c:\PycharmProjects\NPUCM24\train_data_torch.py", line 46, in preprocess
        y_train = torch.from_numpy(y_train)
    TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    ```
    
    这个报错要如何解决？
    
    &nbsp;
    
    GitHub Copilot:
    
    这个错误是因为你试图将一个类型为`numpy.object_`的NumPy数组转换为PyTorch张量。PyTorch只支持将特定类型的NumPy数组转换为张量，包括`float64`、`float32`、`float16`、`complex64`、`complex128`、`int64`、`int32`、`int16`、`int8`、`uint8`和`bool`。
    
    你需要确保`y_train`是这些类型之一。如果`y_train`是类别标签，你可能需要将其转换为整数类型。你可以使用NumPy的`astype`函数来实现这一点：
    
    ```python
    # 数据预处理
    def preprocess(data):
        # ... 代码保持不变 ...
    
        # 将数据转换为PyTorch张量
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train.astype(np.int64))  # 将y_train转换为int64类型
        y_test = torch.from_numpy(y_test.astype(np.int64))  # 将y_test转换为int64类型
    
        return X_train, X_test, y_train, y_test
    ```
    
    这个修改将`y_train`和`y_test`转换为`int64`类型，然后再将它们转换为PyTorch张量。这样，你就可以避免上述错误了。
    
    &nbsp;
    
    SeazSu:
    
    ```bash
    Traceback (most recent call last):
      File "c:\PycharmProjects\NPUCM24\train_data_torch.py", line 114, in <module>
        model = train_model(X_train, y_train, epochs)
      File "c:\PycharmProjects\NPUCM24\train_data_torch.py", line 81, in train_model
        outputs = model(inputs)
      ...
    RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
    ```
    
    这个报错如何解决？
    
    &nbsp;
    
    GitHub Copilot:
    
    这个错误是因为你试图将两个不同数据类型的张量进行矩阵乘法。在你的情况下，一个张量的类型是Double，另一个张量的类型是Float。PyTorch要求这两个张量必须有相同的数据类型。
    
    你可以通过将输入数据转换为Float类型来解决这个问题。你可以在创建张量时使用`.float()`方法来实现这一点：
    
    ```python
    # 数据预处理
    def preprocess(data):
        # ... 代码保持不变 ...
    
        # 将数据转换为PyTorch张量
        X_train = torch.from_numpy(X_train).float()  # 将X_train转换为Float类型
        X_test = torch.from_numpy(X_test).float()  # 将X_test转换为Float类型
        y_train = torch.from_numpy(y_train.astype(np.int64))  # 将y_train转换为int64类型
        y_test = torch.from_numpy(y_test.astype(np.int64))  # 将y_test转换为int64类型
    
        return X_train, X_test, y_train, y_test
    ```
    
    这个修改将`X_train`和`X_test`转换为Float类型，然后再将它们转换为PyTorch张量。这样，你就可以避免上述错误了。

然后运行，发现就可以进行训练了(只是训练的时候没有显示)，

于是(得到权重后)开始编写shap相关的代码，大部分都和 `shap_tf.py` 一样

```python title="shap_torch.py" linenums="1" hl_lines="3 14-15"
import shap
import numpy as np
import torch
from train_data_torch import preprocess, MLP

# read and preprocess data
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, _, _ = preprocess(data)

# select a set of background examples to take an expectation over
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# build model
model = MLP(X_train.shape[1])
model.load_state_dict(torch.load('model_200.pt'))

# Assuming 'model' is your deep learning model and 'background' is a sample of your input data
explainer = shap.DeepExplainer(model, background)

# 'test_data' is the data you want to explain
shap_values = explainer.shap_values(X_test[1:30])

# plot the feature attributions
shap.image_plot(shap_values, -X_test[1:30])
shap.summary_plot(shap_values)
```

进行运行，出现报错

```bash
Traceback (most recent call last):
  File "c:\PycharmProjects\NPUCM24\test3.py", line 28, in <module>
    shap.image_plot(shap_values, X_test)
  File "C:\Users\Ronald\AppData\Local\Programs\Python\Python310\lib\site-packages\shap\plots\_image.py", line 121, in image
    x_curr = x[row].copy()
AttributeError: 'Tensor' object has no attribute 'copy'. Did you mean: 'copy_'?
```

进行询问

???+ example "对话记录"

    RonaldLN:
    
    ```bash
      ...
    AttributeError: 'Tensor' object has no attribute 'copy'. Did you mean: 'copy_'?
    ```
    
    &nbsp;
    
    Github Copilot:
    
    这个错误是因为你试图在PyTorch张量上调用`copy`方法，但是PyTorch张量没有这个方法。你应该使用`clone`方法来复制PyTorch张量。
    
    然而，`shap.image_plot`函数需要的是NumPy数组，而不是PyTorch张量。因此，你需要将PyTorch张量转换回NumPy数组。你可以使用`.numpy()`方法来实现这一点：
    
    ```python
    # plot the feature attributions
    shap.image_plot(shap_values, -X_test.numpy())
    ```
    
    这个修改将`X_test`张量转换回NumPy数组，然后再传递给`shap.image_plot`函数。这样，你就可以避免上述错误了。

感觉又是包的代码的问题，于是继续去修改

```python title=".../plots/_image.py" linenums="121" hl_lines="122"
# x_curr = x[row].copy()
x_curr = x[row].clone()
```

再次运行就可以显示图像了

![shap_summary_plot](../images/shap_summary_plot.svg){ loading=lazy }

## 使用XGBoost模型进行训练，并计算SHAP值

由于使用tensorflow训练的模型尝试了很久都无法成功计算shap值(一直有奇怪的报错)，而用pytorch训练的模型能成功计算shap值但似乎显示的图像也不明所以，

而[官方说明文档中的例子](https://shap.readthedocs.io/en/latest/api_examples.html)中，只有使用XGBoost来训练并计算shap值的例子，

于是打算，先看看官方的例子能不能跑，如果能跑，再把输入训练的数据改成之前处理好的数据(这样应该就不会遇到太多报错)，

于是参考 [`Permutation explainer`](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Permutation.html) 中的例子

??? note "示例代码"

    ```python linenums="1"
    import xgboost
    
    import shap
    
    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    
    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)
    
    # build a Permutation explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Permutation(model.predict_proba, X)
    shap_values = explainer(X[:100])
    
    # get just the explanations for the positive class
    shap_values = shap_values[..., 1]
    
    # Plot a global summary
    shap.plots.bar(shap_values)
    
    # Plot a single instance
    shap.plots.waterfall(shap_values[0])
    ```

能正常运行，然后查看输入数据的格式

>   ```python
>   print(X, "\n", y)
>   print(X.shape, "\n", y.shape)
>   print(type(X), "\n", type(y))
>   ```

```bash
        Age  Workclass  ...  Hours per week  Country
0      39.0          7  ...            40.0       39
1      50.0          6  ...            13.0       39
2      38.0          4  ...            40.0       39
3      53.0          4  ...            40.0       39
4      28.0          4  ...            40.0        5
...     ...        ...  ...             ...      ...
32556  27.0          4  ...            38.0       39
32557  40.0          4  ...            40.0       39
32558  58.0          4  ...            40.0       39
32559  22.0          4  ...            20.0       39
32560  52.0          5  ...            40.0       39

[32561 rows x 12 columns]
 [False False False ... False False  True]
(32561, 12)
 (32561,)
<class 'pandas.core.frame.DataFrame'>
 <class 'numpy.ndarray'>
```

尝试将 `X` 转换成numpy array数组再运行，发现也能运行，只是最后画图时，每个特征没有名字(显示的是 `Feature n` )，于是就开始使用之前处理的数据

```python title="shap_xgboost.py"
...
from train_data import preprocess

# get a dataset on income prediction
# X, y = shap.datasets.adult()
data = np.load('merged_data.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = preprocess(data)

X = X_train
y = y_train

...
```

出现报错

```bash
ValueError: max_evals=500 is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = 26327!
```

于是添加 `max_evals`

```python
explainer = shap.explainers.Permutation(model.predict_proba, X, max_evals=30000)
```

继续运行，出现报错

```bash
Traceback (most recent call last):
  File "c:\PycharmProjects\NPUCM24\shap_xgboost.py", line 28, in <module>
    shap_values = explainer(X[:100])
  ...
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 258. GiB for an array with shape (2632700, 13163) and data type float64  
```

于是在计算SHAP值时，只取5个输入

```python  hl_lines="2-3"
...
X = X_train[:5]
y = y_train[:5]

# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(model.predict_proba, X, max_evals=30000)
shap_values = explainer(X[:100])

...
```

再次运行，就没有报错了，

运行了十几分钟之后，显示出了图像

=== "bar"

    ![shap_bar](../images/shap_bar.svg){ loading=lazy }

=== "waterfall"

    ![shap_waterfall](../images/shap_waterfall.svg){ loading=lazy }

>   可以通过 `max_display` 参数调整显示的特征个数
>
>   >   通过在 `shap/plots/_bar.py` 中查看 `bar` 的定义

---

打算修改XGBoost分类器的设置参数，于是前往官方文档查询API说明

[Python API Reference — xgboost 2.0.3 documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)

由于不清楚参数都是什么意思，于是询问chatgpt

???+ example "对话记录"

    You:
    
    ...(官方文档说明)
    
    &nbsp;
    
    ChatGPT:
    
    当你设置XGBoost分类器时，你可以调整许多参数来控制模型的行为。以下是其中一些主要参数的解释：
    
    1.   **n_estimators**：Boosting轮数，即要构建的基本模型（树）的数量。
    2.   **max_depth**：每棵树的最大深度。它控制树的复杂度，防止过拟合。
    3.   **learning_rate**：学习率，也称为eta。它决定每棵树的贡献程度。较小的学习率需要更多的树来拟合数据，但通常会带来更好的泛化能力。
    4.   **verbosity**：控制XGBoost输出的详细程度。0是最安静的，3是最详细的。
    5.   **objective**：学习任务和相应的学习目标。对于分类任务，默认为'binary:logistic'，表示二分类问题的逻辑回归。
    6.   **booster**：指定使用的增强器类型，可以是'gbtree'（基于树的增强）、'gblinear'（线性模型）或'dart'（Dropouts meet Multiple Additive Regression Trees）。
    7.   **gamma**：叶节点分裂所需的最小损失减少量。它控制树的生长方式，防止过拟合。
    8.   **subsample**：用于训练每棵树的样本子集的比例。它可以防止过拟合，提高模型的泛化能力。
    
    这些是其中一些最重要的参数，根据你的具体问题，可能还有其他参数需要调整。
    
    &nbsp;
    
    You:
    
    在我要训练的数据中，每个输入有13000个特征，你说的这些参数怎么设置更合适？
    
    &nbsp;
    
    ChatGPT:
    
    对于具有13000个特征的数据集，你可能需要特别注意模型的复杂性和训练时间。以下是一些建议的参数设置：
    
    1.   **n_estimators**：通常，你可能需要更多的树来拟合具有13000个特征的数据集。开始时可以尝试较小的值，比如100，然后根据模型的性能逐步增加。
    2.   **max_depth**：由于特征数量较多，你可能需要限制树的深度以防止过拟合。尝试设置一个相对较小的值，比如5或者10。
    3.   **learning_rate**：较小的学习率通常更适合具有大量特征的数据集。开始时可以尝试较小的值，比如0.1或者0.01。
    4.   **subsample**：由于特征数量较多，可以考虑使用较小的样本子集来训练每棵树，以加快训练速度并提高模型的泛化能力。尝试设置一个较小的值，比如0.8。
    5.   **gamma**：可以尝试设置一个较小的值，比如0.1，以控制树的生长方式并防止过拟合。
    6.   **max_bin**：可以考虑增加每个特征的最大bin数以提高模型的拟合能力。
    7.   **colsample_bytree**：考虑使用较小的特征子集来构建每棵树，以增加模型的多样性。尝试设置一个较小的值，比如0.8。
    8.   **objective**：对于分类问题，默认的'binary:logistic'通常是一个不错的选择，但你也可以根据具体情况尝试其他目标函数。
    
    请注意，以上建议是一些常见的经验法则，最终的参数设置应该根据你的数据集和具体问题进行调整和优化。你可能需要使用交叉验证等技术来选择最佳的参数组合。

于是参考chatgpt的回答重新进行训练，

```python title="train_data_xgboost.py" linenums="1"
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
```

(没想到竟然跑了10几分钟)最后显示的预测效果(居然和神经网络的效果差不多，甚至还略好一些)

```bash
              precision    recall  f1-score   support

         0.0       0.96      1.00      0.98        23
         1.0       0.57      0.36      0.44        22
         2.0       0.69      0.83      0.75        35

    accuracy                           0.75        80
   macro avg       0.74      0.73      0.73        80
weighted avg       0.73      0.75      0.73        80
```

---

<h2>代码仓库</h2>

[RonaldLN/NPUCM24](https://github.com/RonaldLN/NPUCM24)
