import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import random


class SCLE(nn.Module):
    def __init__(self, dims, dropout = 0.8):
        super(SCLE, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims) #- 1
        self.alpha = 1.0
        enc = []
        for i in range(self.n_stacks - 1):
            if i == 0:
                enc.append(nn.Dropout(p =dropout))
            enc.append(nn.Linear(self.dims[i], self.dims[i+1]))
            enc.append(nn.BatchNorm1d(self.dims[i+1]))
            enc.append(nn.ReLU())

        enc = enc[:-2]
        self.encoder = nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = torch.nn.functional.normalize(latent_out, dim=1)
        return latent_out


class Classification(nn.Module):
    def __init__(self, input_dim, class_dim):
        super(Classification, self).__init__()
        self.cls = nn.Linear(input_dim, class_dim)

    def forward(self, x):
        out = self.cls(x)
        return out


