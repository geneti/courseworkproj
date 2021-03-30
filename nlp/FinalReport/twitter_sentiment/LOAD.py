import pandas as pd
import csv
import re
from twitter_sentiment.CLEAN import clean
import nltk


def load_data_direct(filename):
    print('Begin loading data')
    very_pos, pos, neutral, neg, very_neg = '','','','',''
    data = pd.read_csv(filename, index_col=0, encoding="ISO-8859-1", dtype=object)
    print(data.iloc[:,3:].head(3))
    for r in range(data.shape[0]):
        if data.iloc[r, 4] == 'Extremely Positive':
            very_pos += data.iloc[r, 3]
        elif data.iloc[r, 4] == 'Positive':
            pos += data.iloc[r, 3]
        elif data.iloc[r, 4] == 'Neutral':
            neutral += data.iloc[r, 3]
        elif data.iloc[r, 4] == 'Negative':
            neg += data.iloc[r, 3]
        elif data.iloc[r, 4] == 'Extremely Negative':
            very_neg += data.iloc[r, 3]
        else:
            print('Error Label, please check data')

    vp = nltk.sent_tokenize(clean(very_pos))
    ps = nltk.sent_tokenize(clean(pos))
    nl = nltk.sent_tokenize(clean(neutral))
    na = nltk.sent_tokenize(clean(neg))
    vn = nltk.sent_tokenize(clean(very_neg))

    return vp, ps, nl, na, vn


def compress_csv(in_file, out_file):
    # print('Begin loading data')
    very_pos, pos, neutral, neg, very_neg = [], [], [], [], []
    data = pd.read_csv(in_file, encoding="ISO-8859-1", dtype=object)
#     print('The shape of dataframe is:', data.shape)
    compress_data = data[['OriginalTweet', 'Sentiment']]
    
    compress_data['OriginalTweet'] = compress_data['OriginalTweet'].str.replace('\s+',' ')
    compress_data['Sentiment'] = compress_data['Sentiment'].str.replace('\s+',' ')
    
    print('The shape of dataframe after compress is:', compress_data.shape)
    # compress_data = data.head(int(data.shape[0] / 100))
#     print(compress_data.head(8))
#     print(compress_data.iloc[:5,0])
    print('Saving compress data')
    compress_data.to_csv(out_file, index = False, encoding='utf-8',line_terminator='\n')

def random_choice_csv(in_file, out_file, n = 0):
    data = pd.read_csv(in_file, encoding="ISO-8859-1", dtype=object)
    data = data[['OriginalTweet', 'Sentiment']]
    data = data.sample(n = n)
    data = data.reset_index(drop=True)
    print(data)
    data.to_csv(out_file, index = False, encoding='utf-8',line_terminator='\n')