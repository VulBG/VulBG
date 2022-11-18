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
import sklearn.model_selection
from metrics import *


device = "cuda:0"
epoches = 50

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

train_log = []
valid_log = []
test_log = []

class TextCNN(nn.Module):
    def __init__(self, fusion = False):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * filter_num, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fusion_fc = nn.Linear(128 + 64, 2)
        self.use_fusion = fusion
    
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

        self.graph_stack = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )

    def forward_baseline(self, baseline_input):
        x = baseline_input
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        cnn_feature = self.cnn_stack(x)
        return self.fc2(cnn_feature)

    def forward_fusion(self, baseline_input, bg_input):
        x = baseline_input
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        cnn_feature = self.cnn_stack(x)
        graph_feature = self.graph_stack(bg_input)
        x = torch.cat((cnn_feature, graph_feature), 1)
        return self.fusion_fc(x)

    def forward(self, baseline_input, bg_input):
        if self.use_fusion:
            return self.forward_fusion(baseline_input, bg_input)
        else:
            return self.forward_baseline(baseline_input, bg_input)
        