import os
import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2(torch.nn.Module):
    def __init__(self, dataset, out_channels, n_hidden_layers):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, out_channels, edge_dim=dataset.num_edge_features)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_conv = []
        for i in range(self.n_hidden_layers):
            self.hidden_conv.append(GATv2Conv(out_channels, out_channels, edge_dim=dataset.num_edge_features))
        self.hidden_conv = ModuleList(self.hidden_conv)
        self.lin = Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, dropout, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        for i in range(self.n_hidden_layers):
            x = self.hidden_conv[i](x, edge_index, edge_attr)
            x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin(x)
        return x