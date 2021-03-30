import pandas as pd
import numpy as np
import math

class dataload(object):
    def __init__(self, address, mean_ordinal = None):
        self.d = pd.read_csv(address, header = 0, index_col = 0)
        # drop features with too many missing values: 
        self.d = self.d.drop(['PoolQC', 'Fence', 'PoolArea'], axis = 1)
        # total sample numbers
        self.sample_num = self.d.shape[0]
        # feature numbers
        self.feature_num = self.d.shape[1]
        # shuffle the data
        self.d = self.d.sample(frac=1).reset_index(drop=True)
        self.mean_ordinal = mean_ordinal

    # Preprocessing the nominal and ordinal data
    def get_data(self):
        # caution: this function only return all raw data with filled missing value
        d1 = self.get_nominal_data()
        d2 = self.get_ordinal_data()
        return pd.concat([d1,d2], axis = 1)

    def get_nominal_data(self):
        nom = self.d.iloc[:,0:self.feature_num-1].select_dtypes(include='object')
        return nom.fillna('missing')
    
    def get_ordinal_data(self):
        ord = self.d.iloc[:,0:self.feature_num-1].select_dtypes(include='float')
        if self.mean_ordinal is not None:
            return ord.fillna(self.mean_ordinal)
        else:
            return ord.fillna(ord.mean())

    def get_label(self):
        return self.d.iloc[:,self.feature_num-1]

    def get_ordinal_mean(self):
        return self.d.iloc[:,0:self.feature_num-1].select_dtypes(include='float').mean()