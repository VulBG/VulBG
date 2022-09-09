import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import sys
import random
import numpy as np 
from numpy import float32
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
import pandas
import sklearn
import torch.nn.functional as F

class CodebertBaseline(nn.Module):
    def __init__(self, fusion = False):
        super(CodebertBaseline, self).__init__()
        self.baseline_fc = nn.Linear(128, 2)
        self.use_fusion = fusion
        
        self.codebert_stack = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )

    def forward_baseline(self, cb_feature, graph_input):
        cb_feature = self.codebert_stack(cb_feature)
        return self.baseline_fc(cb_feature), cb_feature
    
    def forward(self, cb_input, graph_input):
        return self.forward_baseline(cb_input, graph_input)
