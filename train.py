import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import sys
import networkx as nx
import os
import time

from utils import load_data, normalize, cal_accuracy
from model import GCN

datastr = "citeseer"

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(datastr)
#adj = nx.to_scipy_sparse_matrix(adj)
adj_n = normalize(adj)

print(adj.shape, adj_n.shape, features.shape, y_train.shape, train_mask.shape)

embsize = features.shape[1]
n_total = adj.shape[0]
n_train = sum(np.ones(n_total)[train_mask])
n_test = n_total-n_train
nclass = y_train.shape[1]
hidsize = 300
lr = 0.01
weight_decay = 1e-3
print("n_train:",n_train)
norm_adj = normalize(adj)

i = torch.LongTensor([norm_adj.row, norm_adj.col])
v = torch.FloatTensor(norm_adj.data)
norm_adj = torch.sparse.FloatTensor(i,v, adj.shape, requires_grad=False)
features = torch.FloatTensor(features, requires_grad=False)
y_train = torch.FloatTensor(features, requires_grad=False)
y_val = torch.FloatTensor(features, requires_grad=False)
y_test = torch.FloatTensor(features, requires_grad=False)

load_from = False
gcn = GCN(embsize, hidsize, nclass)
if load_from:
	gcn = torch.load(load_from)
optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(200):
	t = time.time()
	gcn.train()
	optimizer.zero_grad()
	output = gcn(features, norm_adj)
	pred = output[train_mask]
	ans = y_train[train_mask]
	loss = F.nll_loss(pred, ans)
	train_acc = cal_accuracy(output, y_train, train_mask)
	loss.backward()
	optimizer.step()

	gcn.eval()
	pred = output[val_mask]
	ans = y_train[val_mask]
	val_loss = F.nll_loss(pred, ans)
	val_acc = cal_accuracy(output, y_val, val_mask)

	print("epoch:", epoch, "time:", time.time()-t)
	print("train_loss:",loss, "train_acc:", train_acc)
	print("val_loss:", val_loss, "val_acc:", val_acc)

torch.save(gcn, "gcn_model")