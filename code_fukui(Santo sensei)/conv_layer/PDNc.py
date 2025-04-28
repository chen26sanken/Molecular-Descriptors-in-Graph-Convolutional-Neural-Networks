import os
import torch
import numpy as np
import csv
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import PDNConv, global_mean_pool, global_max_pool, global_add_pool, aggr
from torch_geometric.nn.norm import LayerNorm, BatchNorm
from pooling import *
from submodules import *

class PDN_dense_add_skip_6bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_concat_after(torch.nn.Module):
    def __init__(self, dataset, device, out_channels, n_hidden_layers, glob_feat_pth):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_concat_after, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.global_feat_path = glob_feat_pth
        with open(self.global_feat_path, "r") as f:
            self.reader = csv.reader(f)
            self.global_feat_set_tmp = [row[2:] for row in self.reader]
            self.global_feat_set = self.global_feat_set_tmp[1:]
            self.global_feat_set = [[s.replace("na", "0") for s in row] for row in self.global_feat_set]
            self.global_feat_set = [list(map(float, row)) for row in self.global_feat_set]
        for v in self.global_feat_set:
            self.size = len(v)
        print(self.out_channels, self.size)
        self.lin = Linear(out_channels+self.size, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = concat_graph_global_feat(x_out, self.global_feat_set, self.device)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_res_add_skip_6bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_res_add_skip_6bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x1
        x = x + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x2
        x = x + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x3
        x = x + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x4
        x = x + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_cat_skip_6bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_cat_skip_6bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.pool_conv_1 = PDNConv(out_channels*2, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.pool_conv_2 = PDNConv(out_channels * 3, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.pool_conv_3 = PDNConv(out_channels * 4, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.pool_conv_4 = PDNConv(out_channels * 5, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.pool_conv_5 = PDNConv(out_channels * 6, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.pool_conv_6 = PDNConv(out_channels * 7, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0), dim=1)
        x = self.pool_conv_1(x, edge_index, edge_attr)
        x1 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0, x1), dim=1)
        x = self.pool_conv_2(x, edge_index, edge_attr)
        x2 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0, x1, x2), dim=1)
        x = self.pool_conv_3(x, edge_index, edge_attr)
        x3 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0, x1, x2, x3), dim=1)
        x = self.pool_conv_4(x, edge_index, edge_attr)
        x4 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0, x1, x2, x3, x4), dim=1)
        x = self.pool_conv_5(x, edge_index, edge_attr)
        x5 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x = torch.cat((x, x0, x1, x2, x3, x4, x5), dim=1)
        x6 = self.pool_conv_6(x, edge_index, edge_attr)
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_supernode(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_supernode, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        print(x, edge_index)
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        print(x, edge_index)

        return 0

        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_concat_after(torch.nn.Module):
    def __init__(self, dataset, device, out_channels, n_hidden_layers, glob_feat_pth):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_concat_after, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.global_feat_path = glob_feat_pth
        with open(self.global_feat_path, "r") as f:
            self.reader = csv.reader(f)
            self.global_feat_set_tmp = [row[2:] for row in self.reader]
            self.global_feat_set = self.global_feat_set_tmp[1:]
            self.global_feat_set = [[s.replace("na", "0") for s in row] for row in self.global_feat_set]
            self.global_feat_set = [list(map(float, row)) for row in self.global_feat_set]
        for v in self.global_feat_set:
            self.size = len(v)
        self.hidden_channels = int((out_channels+self.size) / 2)
        self.lin1 = Linear(out_channels+self.size, self.hidden_channels)
        self.lin2 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = concat_graph_global_feat(x_out, self.global_feat_set, self.device)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_concat_after(torch.nn.Module):
    def __init__(self, dataset, device, out_channels, n_hidden_layers, glob_feat_pth):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_concat_after, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.global_feat_path = glob_feat_pth
        with open(self.global_feat_path, "r") as f:
            self.reader = csv.reader(f)
            self.global_feat_set_tmp = [row[2:] for row in self.reader]
            self.global_feat_set = self.global_feat_set_tmp[1:]
            self.global_feat_set = [[s.replace("na", "0") for s in row] for row in self.global_feat_set]
            self.global_feat_set = [list(map(float, row)) for row in self.global_feat_set]
        for v in self.global_feat_set:
            self.size = len(v)
        self.hidden_channels = int((out_channels+self.size) / 3)
        self.lin1 = Linear(out_channels+self.size, self.hidden_channels*2)
        self.lin2 = Linear(self.hidden_channels*2, self.hidden_channels)
        self.lin3 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_max_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = concat_graph_global_feat(x_out, self.global_feat_set, self.device)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        x_out = x_out.relu()
        x_out = self.lin3(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_1bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_1bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x1 = x1.relu()

        x_out = g_max_pool(x1, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_2bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_2bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x2 = x2.relu()

        x_out = g_max_pool(x2, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_3bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_3bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x3 = x3.relu()

        x_out = g_max_pool(x3, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_4bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_4bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x4 = x4.relu()

        x_out = g_max_pool(x4, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_5bro_pre_batch_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_5bro_pre_batch_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x5 =x5.relu()

        x_out = g_max_pool(x5, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_6bro_original_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_original_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers-1:
                x = x.relu()
        x1 = x + x0
        x1 = x1.relu()
        x = x + x0
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers - 1:
                x = x.relu()
        x2 = x + x0 + x1
        x2 = x2.relu()
        x = x + x0 + x1
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers - 1:
                x = x.relu()
        x3 = x + x0 + x1 + x2
        x3 = x3.relu()
        x = x + x0 + x1 + x2
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers - 1:
                x = x.relu()
        x4 = x + x0 + x1 + x2 + x3
        x4 = x4.relu()
        x = x + x0 + x1 + x2 + x3
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers - 1:
                x = x.relu()
        x5 = x + x0 + x1 + x2 + x3 + x4
        x5 = x5.relu()
        x = x + x0 + x1 + x2 + x3 + x4
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
            if i < self.n_hidden_layers - 1:
                x = x.relu()
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_relu_act_max_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_relu_act_max_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
            x = Norm_layer(x)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_v1(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_v1, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.hidden_channels = 16
        self.lin1 = Linear(out_channels, self.hidden_channels)
        self.lin2 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v1(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v1, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.hidden_channels = 16
        self.lin1 = Linear(out_channels, self.hidden_channels)
        self.lin2 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin3 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        x_out = x_out.relu()
        x_out = self.lin3(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_v2(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_v2, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.hidden_channels = 32
        self.lin1 = Linear(out_channels, self.hidden_channels)
        self.lin2 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v2(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v2, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.hidden_channels = 16
        self.lin1 = Linear(out_channels, self.hidden_channels*2)
        self.lin2 = Linear(self.hidden_channels*2, self.hidden_channels)
        self.lin3 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        x_out = x_out.relu()
        x_out = self.lin3(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v3(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_max_test_3lin_v3, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features,
                             dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.hidden_channels = 32
        self.lin1 = Linear(out_channels, self.hidden_channels)
        self.lin2 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin3 = Linear(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = g_max_pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin1(x_out)
        x_out = x_out.relu()
        x_out = self.lin2(x_out)
        x_out = x_out.relu()
        x_out = self.lin3(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_mean_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_mean_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_mean_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_sum_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_sum_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()
        # print(x6.size())

        x_out = g_add_pool(x6, batch)
        # print(x_out.size())

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        # print(x_out.size())
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_set_aggr_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_set_aggr_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(self.out_channels*2, dataset.num_classes)
        self.pool = aggr.Set2Set(in_channels=self.out_channels, processing_steps=5)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = self.pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out

class PDN_dense_add_skip_6bro_pre_batch_act_attn_aggr_test(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(PDN_dense_add_skip_6bro_pre_batch_act_attn_aggr_test, self).__init__()
        self.out_channels = out_channels
        self.conv1 = PDNConv(dataset.num_node_features, out_channels, dataset.num_edge_features, dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv_1 = []
        self.hidden_conv_2 = []
        self.hidden_conv_3 = []
        self.hidden_conv_4 = []
        self.hidden_conv_5 = []
        self.hidden_conv_6 = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv_1.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_2.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_3.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_4.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_5.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
            self.hidden_conv_6.append(
                PDNConv(out_channels, out_channels, dataset.num_edge_features, dataset.num_edge_features))
        self.hidden_conv_1 = ModuleList(self.hidden_conv_1)
        self.hidden_conv_2 = ModuleList(self.hidden_conv_2)
        self.hidden_conv_3 = ModuleList(self.hidden_conv_3)
        self.hidden_conv_4 = ModuleList(self.hidden_conv_4)
        self.hidden_conv_5 = ModuleList(self.hidden_conv_5)
        self.hidden_conv_6 = ModuleList(self.hidden_conv_6)
        self.lin = Linear(out_channels, dataset.num_classes)
        self.gate_nn = gate_model(out_channels)
        self.pool = aggr.AttentionalAggregation(gate_nn=self.gate_nn)

    def forward(self, x, edge_index, batch, dropout, edge_attr, device):
        Norm_layer = BatchNorm(self.out_channels).to(device)
        x = self.conv1(x, edge_index, edge_attr)
        x0 = x
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_1[i](x, edge_index, edge_attr)
        x1 = x + x0
        x = x + x0
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_2[i](x, edge_index, edge_attr)
        x2 = x + x0 + x1
        x = x + x0 + x1
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_3[i](x, edge_index, edge_attr)
        x3 = x + x0 + x1 + x2
        x = x + x0 + x1 + x2
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_4[i](x, edge_index, edge_attr)
        x4 = x + x0 + x1 + x2 + x3
        x = x + x0 + x1 + x2 + x3
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_5[i](x, edge_index, edge_attr)
        x5 = x + x0 + x1 + x2 + x3 + x4
        x = x + x0 + x1 + x2 + x3 + x4
        for i in range(self.n_hidden_layers):
            x = Norm_layer(x)
            x = x.relu()
            x = self.hidden_conv_6[i](x, edge_index, edge_attr)
        x6 = x + x0 + x1 + x2 + x3 + x4 + x5
        x6 = x6.relu()

        x_out = self.pool(x6, batch)

        x_out = F.dropout(x_out, p=dropout, training=self.training)
        x_out = self.lin(x_out)
        return x_out