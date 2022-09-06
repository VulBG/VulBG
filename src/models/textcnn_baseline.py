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

filter_sizes = [3,4,5]
channel_num = 1
filter_num = 100
embedding_dimension = 100
dropout = 0.3

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * filter_num, 128)
        self.fc2 = nn.Linear(128, 2)
    
        self.cnn_stack = nn.Sequential(
            nn.Linear(len(filter_sizes) * filter_num, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )

        # self.graph_stack = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.3),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.3),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.3),
        # )

    def forward(self, x, graph_input):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        cnn_feature = self.cnn_stack(x)
        # x = cnn_feature
        # graph_feature = self.graph_stack(graph_input)
        # x = torch.cat((cnn_feature, graph_feature), 1)
        # x = self.fc2(x)
        return self.fc2(cnn_feature), cnn_feature



# def load_graph(path, n, final_data, tag):
#     f = open("%s/embed_%s_%s.txt" % (path, tag, n), "r").read()
#     node_vecs = {}
#     miss = 0
#     f = f.split("\n")
#     print(len(f))
#     for line in f[1:]:
#         if not line:
#             continue
#         line = line.split(" ")
#         node_id = int(line[0])
#         vec = [float(i) for i in line[1:]]
#         node_vecs[node_id] = np.array(vec)#, dtype="float32")
        
#     func_idx_begin = n
#     for func in final_data:
#         if func_idx_begin not in node_vecs:
#             func["graph_vec"] = np.zeros(128)
#             print(func_idx_begin)
#             print("zero!")
#         else:
#             func["graph_vec"] = node_vecs[func_idx_begin]
#         func_idx_begin += 1

# if __name__ == "__main__":
#     tag = sys.argv[1]
#     n = sys.argv[2]
#     if tag == "reveal":
#         path = "/root/data/VulBG/dataset/devign_dataset.pkl"
#     elif tag == "devign":
#         path = "/root/data/VulBG/dataset/reveal_dataset.pkl"

#     dataset = pickle.load(open(path, "rb"))
#     #load_graph("/root/data/VulBG/dataset/embeds/", n, dataset, tag)

