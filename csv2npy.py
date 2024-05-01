import pandas as pd
import numpy as np

# 将csv文件中的数据读取成numpy数组，并保存为npy文件
def csv2npy(csv_file, npy_file):
    data = pd.read_csv(csv_file)
    data = data.values
    np.save(npy_file, data)


if __name__ == '__main__':
    names = ['附件1：Normal_exp', '附件2：EarlyStage_exp', '附件3：LaterStage_exp']
    for n in names:
        # csv2npy(n + '.csv', n + '.npy')
        data = pd.read_csv(n + '.csv')
        print(data.values.shape)

    # data = pd.read_csv('附件1：Normal_exp.csv')
    # print(data.values)
