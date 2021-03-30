import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

raw_data = pd.read_csv('./raw_data.csv', header = 0, index_col = 0)
sample_num = raw_data.shape[0]

# sort features by nominal or non-nominal 
dtypes = {}
for j in range(raw_data.shape[1]):
    if isinstance(raw_data.iloc[0,j], str) or pd.isna(raw_data.iloc[0,j]):
        dtypes[raw_data.columns[j]] = str
    else:
        dtypes[raw_data.columns[j]] = np.float64

data = pd.read_csv('./raw_data.csv',sep = ',', header = 0, index_col = 0, dtype = dtypes)

# separate the housing prices into several zones
data['PriceLevel'] = 'level'

for i in range(sample_num):
    if data.iloc[i,79] <= 135000:
        data.iloc[i,80] = 'level_1'
    elif data.iloc[i,79] <= 165000:
        data.iloc[i,80] = 'level_2'
    elif data.iloc[i,79] <= 200000:
        data.iloc[i,80] = 'level_3'
    else:
        data.iloc[i,80] = 'level_4'
data = data.drop(columns = 'SalePrice')

#shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
print('data: ',data)


tmp = sample_num*9/10
print(data.shape)
train = data.iloc[0:int(tmp),:]
test = data.iloc[int(tmp)+1:sample_num,:]

train.to_csv('./train.csv')
test.to_csv('./test.csv')