import pickle
import random 
import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def derandom():
    seed = 202203
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

final_data = pickle.load(open("/root/ase2022/dataset/final_slice_data.pkl", "rb"))
train_data_idx, test_data_idx = pickle.load(open("/root/ase2022/dataset/data_split.pkl", "rb"))

np_final_data = np.array(final_data)
train_data = np_final_data[train_data_idx]
test_data = np_final_data[test_data_idx]

all_embeds = []
train_embeds = []
for i in final_data:
    all_embeds += i["vec"]
for i in train_data:
    train_embeds += i["vec"]

all_embeds_scaled = [i * 1000 for i in all_embeds]
train_embeds_scaled = [i * 1000 for i in train_embeds]


def train_kmeans(n, inputs):
    kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, verbose=1, max_iter=300, batch_size=3000)
    for i in range(0, len(inputs), 3000):
        kmeans.partial_fit(inputs[i:i+3000])
    return kmeans

# def coloring_node(n, vec_inputs, labels, cluster_model):
#     color_weight = {}
#     for i in range(n):
#         color_weight[i] = 0
#     preds = cluster_model.predict(vec_inputs)
#     for i in range(len(vec_inputs)):
#         if labels[i] == 1:
#             color_weight[preds[i]] += 1
#         else:
#             color_weight[preds[i]] -= 1
#     return color_weight


def generate_edges(n, final_data, cluster_model, node_weights = None, use_dis = False):
    edges = []
    func_idx_begin = n
    for func in final_data:
        tmp_vec = [i * 1000 for i in func["vec"]]
        tmp_labs = cluster_model.predict(tmp_vec)

        for i in range(len(tmp_vec)):
            lab = tmp_labs[i]
            d = np.linalg.norm(cluster_model.cluster_centers_[lab] - (tmp_vec[i]))
            if node_weights or use_dis:
                if use_dis:
                    edges.append([func_idx_begin, lab, 10000/(d+1)])
                else:
                    edges.append([func_idx_begin, lab, node_weights[lab]])
            else:
                edges.append([func_idx_begin, lab])
        func_idx_begin += 1
    return edges
        
def save_edges(edges, fname, have_weight = False):
    f = open(fname, "w")
    for i in edges:
        if have_weight:
            f.write(str(i[0]) + " " + str(i[1]) +" " + str(i[2]) +"\n")
        else:
            f.write(str(i[0]) + " " + str(i[1]) + "\n")
            
    f.close()
    print("Done")
            
def do_embed(edges_fname, output_fname, have_weight = False):
    cmdline = "/root/data/ase2022/src/snap/examples/node2vec/node2vec -i:%s -o:%s -e:2"
    if have_weight:
        cmdline += "-w"

    ret = os.system(cmdline % (edges_fname, output_fname))
    print("node2vec returns %d" % ret)
        

def gen_graph(n, inputs, final_data, tag):
    kmeans = train_kmeans(n, inputs)
#     node_weights = coloring_node(n, inputs, labels, kmeans)
    edges = generate_edges(n, final_data, kmeans, None, use_dis=True)
    print(len(edges))
    save_edges(edges, "edges_%s.txt" % str(tag), True)
    do_embed("edges_%s.txt" % str(tag), "embed_%s.txt" % str(tag), True)
    
def load_graph(n, final_data, tag):
    f = open("embed_%s.txt" % str(tag), "r").read()
    node_vecs = {}
    miss = 0
    f = f.split("\n")
    for line in f[1:]:
        if not line:
            continue
        line = line.split(" ")
        node_id = int(line[0])
        vec = [float(i) for i in line[1:]]
        node_vecs[node_id] = np.array(vec)#, dtype="float32")
        
    all_graph_data = []
    func_idx_begin = n
    for func in final_data:
        if func_idx_begin not in node_vecs:
            func["graph_vec"] = np.zeros(128)
            print(func_idx_begin)
            print("zero!")
        else:
            func["graph_vec"] = node_vecs[func_idx_begin]
        func_idx_begin += 1

def mycollate_fn(data):
    data.sort(key=lambda x: len(x["word2vec"]), reverse=True)  # 根据input来排序
    data_length = 0
    input_data = []
    label_data = []
    graph_data = []
    for i in data:
        label_data.append(i["label"])
        graph_data.append(i["graph_vec"])
    label_data = torch.tensor(label_data)
    graph_data = np.array(graph_data)
    graph_data = torch.tensor(graph_data, dtype=torch.float32)
    return input_data, label_data, graph_data,data_length


from torch import nn
import torch
from torch.utils.data import DataLoader
from metrics import *


device = "cuda:0"
good_n = {}
class graph_nn(nn.Module):
    def __init__(self):
        super(graph_nn, self).__init__()
        self.linear_stack2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        logits = self.linear_stack2(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, verbose = 0):
    all_labs = []
    all_preds = []
    model.train()
    
    train_loss = 0
    num_batchs = len(dataloader)
 
    for i, data in enumerate(dataloader, 0):  # 0是下标起始位置默认为0
        y, X =  data[1].to(device), data[2].to(device)
        X = X.to(torch.float)
        X = X.to(device)
        pred = model(X)
        # print(pred)
#         y = y.to(torch.float32)
        y = y.to(device)
        loss = loss_fn(pred, y)
        train_loss += loss

        # stats
        _,pred_labs = pred.topk(1)
        for pl in pred_labs:
            all_preds.append(pl.item())
        for yy in y:
            all_labs.append(yy.item())

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        del pred
        del y
        torch.cuda.empty_cache()

    train_loss /= num_batchs
    # print(all_labs)
    # print(all_preds)
  
    scores = get_scores(all_labs, all_preds)
    if verbose:
        print(f'Train: {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')


    
    
def test_loop(dataloader, nn_model, loss_fn, use_pooler = True, verbose = 0, is_valid = 0):
    nn_model.eval()
    all_labs = []
    all_preds = []
    
    test_loss = 0
    num_batchs = len(dataloader)

        
    for i, data in enumerate(dataloader, 0):  # 0是下标起始位置默认为0
        y, X =  data[1].to(device), data[2].to(device)
        X = X.to(torch.float)
        pred = nn_model(X)
        y = y.to(device)
        loss = loss_fn(pred, y)
        test_loss += loss

        # stats
        _,pred_labs = pred.topk(1)
        for pl in pred_labs:
            all_preds.append(pl.item())
        for yy in y:
            all_labs.append(yy.item())

        del pred
        del y
        torch.cuda.empty_cache()

    test_loss /= num_batchs
    
    scores = get_scores(all_labs, all_preds)
#     print(scores)
    if verbose:
        if is_valid:
            print(f'Valid: {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')
        else:
            print(f'Test : {(100*scores[0]):>0.2f}%, {(100*scores[1]):>0.2f}%, {(100*scores[2]):>0.2f}%, {(100*scores[3]):>0.2f}%')

    return 100*scores[0]

     
def train_a_nn(n, final_data, verbose = 0):
    max_f1 = 0
    max_valid_score = 0
    nn_model = graph_nn()
    nn_model.to(device)   
    batch_size = 32
    learning_rate = 3e-4
#     loss_fn = nn.BCEWithLogitsLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
    
    list_data = []
    for func in final_data:
        list_data.append((func["graph_vec"], func["label"]))
        
    np_final_data = np.array(final_data)

    for train_idx, test_idx in kf.split(final_data):
        train_data = np_final_data[train_idx]
        test_data = np_final_data[test_idx]
        test_data = np.array(test_data)
    #     print(test_data)
        valid_data = test_data[len(test_data)>>1:]
        test_data = test_data[:len(test_data)>>1]
        break
        
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False, collate_fn=mycollate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = False, collate_fn=mycollate_fn)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False, collate_fn=mycollate_fn)
    
    torch.cuda.empty_cache()
    for t in range(100):
#         print(f"Epoch {t+1}-------------------------------")
        train_loop(train_dataloader, nn_model, loss_fn, optimizer, verbose=verbose)
        valid_score = test_loop(valid_dataloader, nn_model, loss_fn, verbose=verbose, is_valid=1)
        f1 = test_loop(test_dataloader, nn_model, loss_fn, verbose=verbose)
        if valid_score > max_valid_score:
            max_valid_score = valid_score
            if f1 > max_f1:
                max_f1 = f1
            print(f'Valid: {(100*valid_score):>0.2f}%, Test : {(100*f1):>0.2f}%')
        
    print("Done! %d" % n)
    good_n[n] = max_f1
    return nn_model

import sys
if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"usage: {(__file__):s} cluster_center_num")
        exit(0)
    n = sys.argv[1]
    gen_graph(n, train_embeds_scaled, final_data, n)
    load_graph(n, final_data, i)
    train_a_nn(n, final_data, n)
