import numpy as np
import pandas as pd

headers = pd.read_csv('附件1：Normal_exp.csv')
headers = np.array(headers)

# get the first column as header
headers = headers[:, 0]
print(headers)
print(headers.shape)

# save the headers to npy file
np.save('headers.npy', headers)
