import numpy as np
import torch.nn as nn

def gate_model(datasize):
    model = nn.Sequential(
            nn.Linear(datasize, datasize),
            nn.ReLU(),
            nn.Linear(datasize, datasize),
            nn.ReLU(),
            nn.Linear(datasize, 1),
            nn.ReLU()
            )

    return model