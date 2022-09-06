from gensim.models import word2vec
import torch
import pickle
import os
import sys
import random
import gensim
from torch import nn 
import numpy as np 
from numpy import float32
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import pandas
import sklearn
import torch.nn.functional as F

seed = 202203
random.seed(seed)
os.environ['PYHTONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == "reveal":
        path = "/root/data/VulBG/dataset/devign_dataset.pkl"
    elif dataset == "devign":
        path = "/root/data/VulBG/dataset/reveal_dataset.pkl"

    dataset = pickle.load(open(path, "rb"))

    train_data, test_data = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 0, shuffle = True)

    input_list = []
    for train_data_ in train_data:
        code = train_data_['code']
        codes = code.split("\n")
        for line in codes:
            line = line.split()

    model = word2vec.Word2Vec(input_list, min_count=5)
    for data in dataset:
        code = data['code']
    codes = code.split()
    if(len(codes) > 768):
        codes = codes[:768]

    embeddings = []
    for c in codes:
        try:
            embedding = model[c]
        except:
            zeros = [0 for i in range(100)]
            embedding = pandas.array(data=zeros,dtype=float32)
        embeddings.append(embedding)
    data['word2vec'] = embeddings
    pickle.dump(dataset, open(path, "wb"))
    print("word2vec done")






