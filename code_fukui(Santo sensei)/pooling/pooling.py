import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import aggr

def g_mean_pool(x, batch):
    pool = aggr.MeanAggregation()
    return pool(x, batch)

def g_max_pool(x, batch):
    pool = aggr.MaxAggregation()
    return pool(x, batch)

def g_add_pool(x, batch):
    pool = aggr.SumAggregation()
    return pool(x, batch)

# def g_set_pool(x, batch):
#     pool = aggr.Set2Set(in_channels=, processing_steps=)
#     return pool
#
# def g_attn_pool(x, batch):
#     pool = aggr.AttentionalAggregation