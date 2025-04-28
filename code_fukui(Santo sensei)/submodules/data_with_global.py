import os
import random

import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from termcolor import colored
import csv

class Use_global_feat:
    def __init__(self, global_feat_path):
        self.global_feat_path = global_feat_path
        with open(self.global_feat_path, "r") as f:
            self.reader = csv.reader(f)
            self.global_feat_set_tmp = [row[2:] for row in self.reader]
            self.global_feat_set = self.global_feat_set_tmp[1:]
            self.global_feat_set = [[s.replace("na", "0") for s in row] for row in self.global_feat_set]
            self.global_feat_set = [list(map(float, row)) for row in self.global_feat_set]

class Concat_node_global_feat(Use_global_feat):
    def __init__(self, global_feat_path):
        super().__init__(global_feat_path)

    def __call__(self, data):
        global_feat = self.global_feat_set.pop(0)
        size = data.x.size()[0]
        a = torch.zeros(size, len(global_feat), dtype=torch.float)
        b = torch.tensor(global_feat)
        c = torch.add(a, b)
        d = torch.cat((data.x, c), dim=1)
        data.update({"x":d})
        return data

def concat_graph_global_feat(graph_feat, global_feat_set, device):
    for n in range(graph_feat.size()[0]):
        if n == 0:
            global_feat = global_feat_set.pop(0)
            global_feat_set.append(global_feat)
            global_feat_tr = torch.tensor(global_feat).to(device)
            global_feat_tr = torch.reshape(global_feat_tr, (1, -1))
        else:
            global_feat_tmp = global_feat_set.pop(0)
            global_feat_set.append(global_feat_tmp)
            global_feat_tmp2 = torch.tensor(global_feat_tmp).to(device)
            global_feat_tmp2 = torch.reshape(global_feat_tmp2, (1, -1))
            global_feat_tr = torch.cat((global_feat_tr, global_feat_tmp2), dim=0)
    # global_feat_tr = torch.tensor(global_feat).to(device)
    # print(graph_feat.size(), global_feat_tr.size())
    re_graph_feat = torch.cat((graph_feat, global_feat_tr), dim=1)
    return re_graph_feat

class Add_node_global_feat(Use_global_feat):
    def __init__(self, global_feat_path):
        super().__init__(global_feat_path)

    def __call__(self, data):
        global_feat = self.global_feat_set.pop(0)
        #add global feature as a node to node feature
        size_node = data.x.size()[0]
        size_node_feat = data.x.size()[1]
        a = torch.zeros(size_node, len(global_feat), dtype=torch.float)
        b = torch.zeros(1, size_node_feat, dtype=torch.float)
        c = torch.tensor(global_feat)
        c = torch.reshape(c, (1, c.size()[0]))
        d = torch.cat((data.x, a), dim=1)
        e = torch.cat((b, c), dim=1)
        new_node_feat = torch.cat((d, e), dim=0)

        #add edges that are from global node to other nodes
        size_edge = data.edge_attr.size()[0]
        size_edge_feat = data.edge_attr.size()[1]
        a = torch.zeros(size_edge, 1, dtype=torch.float)
        b = torch.zeros(size_node, size_edge_feat, dtype=torch.float)
        c = torch.ones(size_node, 1, dtype=torch.float)
        d = torch.cat((data.edge_attr, a), dim=1)
        e = torch.cat((b, c), dim=1)
        new_edge_feat = torch.cat((d, e), dim=0)

        #add adjacency matrix asdirected graph
        a = torch.tensor(np.array([size_node for i in range(size_node)]))
        b = torch.tensor(np.array([i for i in range(size_node)]))
        c = torch.stack((a, b), dim=0)
        new_edge_index = torch.cat((data.edge_index, c), dim=1)

        data.update({"x":new_node_feat, "edge_attr":new_edge_feat, "edge_index":new_edge_index})
        return data