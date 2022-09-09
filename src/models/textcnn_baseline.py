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


def train(train_loader, device, model, criterion, optimizer):
    model.train()
    num_batch = len(train_loader)
    model = model.to(device)
    train_loss = 0.0
    all_labs = []
    all_preds = []
    for i, data in enumerate(train_loader, 0):  
        inputs, labels, graph_input, batch_seq_len,  = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        optimizer.zero_grad()
        outputs, last_hidden = model(inputs,graph_input)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _,pred = outputs.topk(1)

        train_loss += loss.item()
        for mm in labels:
            all_labs.append(mm.item())
        for mm in pred:
            all_preds.append(mm.item())
    
    scores = get_scores(all_labs, all_preds) 
    train_loss/= num_batch
    print(f'Train: {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')
    train_log.append(scores)

def valid(validate_loader, device, model, criterion, is_valid=True):
    model = model.to(device)
    model.eval()
    num_batch = len(validate_loader)
    model = model.to(device)
    test_loss = 0.0
    all_labs = []
    all_preds = []
    for i, data in enumerate(validate_loader, 0): 
        inputs, labels, graph_input, batch_seq_len =  data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        outputs, last_hidden = model(inputs, graph_input)

        loss = criterion(outputs,labels)
        _,pred = outputs.topk(1)
        test_loss += loss.item()
        for mm in labels:
            all_labs.append(mm.item())
        for mm in pred:
            all_preds.append(mm.item())
    
    scores = get_scores(all_labs, all_preds) 
    test_loss /= num_batch
    if is_valid:
        valid_log.append(scores)
        print(f'Valid: {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')
    else:
        test_log.append(scores)
        print(f'Test : {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')
    return scores

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

def mycollate_fn(data):
    # 这里的data是getittem返回的（input，label）的二元组，总共有batch_size个
    data.sort(key=lambda x: len(x["word2vec"]), reverse=True)  # 根据input来排序
    data_length = [len(sq["word2vec"]) for sq in data]
    input_data = []
    label_data = []
    graph_data = []
    for i in data:
        input_data.append(torch.tensor(np.array(i["word2vec"])))
        label_data.append(i["vul"])
        graph_data.append(i["graph_vec"])
    input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
    label_data = torch.tensor(label_data)
    graph_data = np.array(graph_data)
    graph_data = torch.tensor(graph_data, dtype=torch.float32)
    return input_data, label_data, graph_data,data_length


if __name__ == "__main__":
    #path = "/root/data/VulBG/dataset/devign_dataset.pkl"
    path = sys.argv[1]
    dataset = pickle.load(open(path, "rb"))
    #load_graph("/root/data/VulBG/dataset/embeds/", 1150, dataset, "devign")
    model = TextCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train_data, test_data = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 0, shuffle = True)
    valid_data, test_data = sklearn.model_selection.train_test_split(test_data, test_size = 0.5, random_state = 0, shuffle = True)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False, collate_fn=mycollate_fn,)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, collate_fn=mycollate_fn,)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=mycollate_fn,)

    for i in range(epoches):
        print("----epoch %d----" % i)
        train(train_loader, device, model, criterion, optimizer)
        valid(valid_loader, device, model, criterion, True)
        valid(test_loader, device, model, criterion, False)

        


