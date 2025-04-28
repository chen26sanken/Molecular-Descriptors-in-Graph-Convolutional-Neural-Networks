import os
import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, n_hidden_layers):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv.append(GCNConv(hidden_channels, hidden_channels))
        self.hidden_conv = ModuleList(self.hidden_conv)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv[i](x, edge_index, edge_attr)
            x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin(x)
        return x