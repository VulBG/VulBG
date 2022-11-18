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

class CodebertMixin(nn.Module):
    def __init__(self, fusion = False):
        super(CodebertMixin, self).__init__()
        self.baseline_fc = nn.Linear(128, 2)
        self.fusion_fc = nn.Linear(128 + 64, 2)
        self.use_fusion = fusion
        
        self.codebert_stack = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )
        
        self.bg_stack = nn.Sequential(
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

    def forward_baseline(self, baseline_input, bg_input):
        baseline_input = self.codebert_stack(baseline_input)
        return self.baseline_fc(baseline_input)

    def forward_fusion(self, baseline_input, bg_input):
        baseline_input = self.codebert_stack(baseline_input)
        graph_feature = self.graph_stack(bg_input)
        fusion = torch.cat((cb_feature, graph_feature), 1)
        return self.fusion_fc(fusion)

    def forward(self, baseline_input, bg_input):
        if self.use_fusion:
            return self.forward_fusion(baseline_input, bg_input)
        else:
            return self.forward_baseline(baseline_input, bg_input)
