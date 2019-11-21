import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import sys
import networkx as nx
import os
from utils import load_data, normalize
from model import GCN

datastr = "citeseer"

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(datastr)
adj = nx.to_scipy_sparse_matrix(adj)
adj_n = normalize(adj)

print(adj_n.shape, features.size(), y_train.size(), train_mask.size())