import timeit
import importlib
import gensim
import nltk
import json
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from twitter_sentiment import utils


class LSTM(nn.Module):
    
    #define all the layers used in model
    
    # Whenever an instance of a class is created, init function is automatically invoked. 
    # We will define all the layers that we will be using in the model
    def __init__(self, vocab_size, embedding_dim, embeddings, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        
        super().__init__()          
        
        #embedding layer
        self.embedding        = nn.Embedding(vocab_size, embedding_dim)  # This is a simple lookup table that turns the one-hot encodings into a vector - just like word2vec!
        self.embedding.weight = nn.Parameter(embeddings)                 # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,                        # Dimension of input
                            hidden_dim,                           # Number of hidden nodes
                            num_layers    = n_layers,             # Number of layers to be stacked
                            bidirectional = bidirectional,        # If True, uses a Bi directional LSTM
                            dropout       = dropout               # Introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
                           )
        
        # dense layer
        multiplier = 1 if bidirectional == False else 2
        self.fc    = nn.Linear(hidden_dim * multiplier, output_dim)
        
        # final activation function
        self.act = nn.Sigmoid()
    
    # This is where we chain together to layers to get the output we want
    def forward(self, text, text_lengths, bidirectional):
        
        # 1. Embedding: is used to convert the one-hot word representations into their
        #               Word vector representations. 
        embedded = self.embedding(text)
      
        # 2. Pack padding: tells the network to ignore the inputs that are `<pad>`. 
        #                  so that the outputs we generate are not influenced by the `<pad>` tokens; 
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        
        # 3. LSTM: is a variant of RNN that captures long term dependencies.
        #          Following some important parameters of LSTM that you should be familiar with. Given below are the parameters of this layer:Run the LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
      
        # 4. Concatenate: the final forward and backward hidden state
        hidden = hidden[-1,:,:] if bidirectional == False else torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        # 5. Linear: Connect to a densely connected linear layer
        dense_outputs =self.fc(hidden)
            

        # 6. ACtivation: Pass value to our sigmoid 
        outputs=self.act(dense_outputs)
        
        return outputs

class AllData(data.Dataset):
    def __init__(self,X,y,transform=None, target_transform=None):
        self.data = X
        self.targets = y
        self.transform = transform
        self.target_transform= target_transform
    def __getitem__(self, index):
        item, target = self.data[index], self.targets[index]
        item = torch.FloatTensor(item)
        target = torch.LongTensor([target])
        return item, target

    def __len__(self):
        return len(self.data)
    
def getTensorBatch(data_path, batch_number, batch_size, random_seed, max_sequence_length, vocabulary):
#     print(data_path)
    #-----------------------------------------------------
    # Import a batch of data
    #-----------------------------------------------------
    data, end_flag = utils.getBatch(data_path,  batch_number, batch_size, random_seed)

    #-----------------------------------------------------
    # Unpack the JSON
    #-----------------------------------------------------
    reviews,freshness  = [], []
    for row in data:
        reviews.append(row['OriginalTweet'])
        freshness.append(row['Sentiment'])
    freshness = [x for x in freshness]
    tmp = []
    for x in freshness:
        if x == "Extremely Positive":
            tmp.append(4)
        elif x == "Positive":
            tmp.append(3)
        elif x == "Neutral":
            tmp.append(2)
        elif x == "Negative":
            tmp.append(1)
        else:
            tmp.append(0)

    #-----------------------------------------------------
    # Format the outcome tensor - y
    #-----------------------------------------------------
    y = torch.tensor(tmp)
    y = y.to(torch.float)
    y = y.view(np.shape(y)[0] , 1)

    #-----------------------------------------------------
    # Format the input tensor - X
    #-----------------------------------------------------
    for i,sentence in enumerate(reviews):
        reviews[i] = nltk.word_tokenize(gensim.utils.to_unicode(sentence.lower()))
    onehot_sentences = utils.onehot(list_of_tokenized_sentences = reviews, vocabulary = vocabulary)

    seq_lengths = torch.LongTensor(list(map(len, onehot_sentences)))
    seq_tensor  = Variable(torch.zeros((len(onehot_sentences), max_sequence_length))).long()

    for idx, (seq, seqlen) in enumerate(zip(onehot_sentences, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    seq_tensor = seq_tensor.transpose(0,1)

    return seq_tensor, seq_lengths, y, end_flag
    
def train(epoch, model, optimizer, criterion, bidirectional, Vocab):
    torch.manual_seed(np.random.randint(1, 100))
    batch_idx = 0
    end_flag = True
    train_loss = 0
    model.train()
    correct = 0
    total = 0
    while end_flag!=False:
        optimizer.zero_grad()
        data_path     = 'twitter_sentiment/train_simple.jsonl'
        X, X_lengths, y, end_flag = getTensorBatch(data_path, 
                                           batch_number        = batch_idx, 
                                           batch_size          = 100,
                                           random_seed         = np.random.randint(1, 100), 
                                           max_sequence_length = 100,
                                            vocabulary         = Vocab)
        outputs = model(X, X_lengths, bidirectional)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    if epoch% 10==0:
        np.set_printoptions(precision=4)
        print(epoch, ' train loss:', math.exp(train_loss))
    return train_loss, float(correct)/total

# Define the test function
def test(epoch, model, optimizer, criterion, bidirectional, Vocab):
    torch.manual_seed(np.random.randint(1, 100))
    batch_idx = 0
    end_flag = True
    test_loss = 0
    correct = 0
    total = 0
    
    model.eval()

    with torch.no_grad():
        while end_flag!=False:
            data_path     = 'twitter_sentiment/test_simple.jsonl'
            X, X_lengths, y, end_flag = getTensorBatch(data_path, 
                                           batch_number        = batch_idx, 
                                           batch_size          = 100,
                                           random_seed         = np.random.randint(1, 100), 
                                           max_sequence_length = 100,
                                            vocabulary         = Vocab)
            outputs = model(X, X_lengths, bidirectional)
            loss = criterion(outputs, y)
                
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    if epoch% 10==0:
        np.set_printoptions(precision=4)
        print(epoch, ' test loss:', math.exp(test_loss))
    return test_loss, float(correct)/total