import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

num_bins = 100

raw_data = pd.read_csv('./raw_data.csv', header = 0, index_col = 0)
sample_num = raw_data.shape[0]
print(sample_num)
label = raw_data.iloc[:,raw_data.shape[1]-1]
price = label.values
print('max price: ', max(price))
print('min price: ', min(price))
df = raw_data.loc[raw_data['SalePrice'] <= 135000]
print(df.shape[0])
plt.hist(price, num_bins)
plt.show()