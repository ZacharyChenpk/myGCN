import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F

class singleGCNLayer(nn.Module):

	def __init__(self, embsize, outsize, nolinear="NO"):
		super(RenormalizeGCNLayer, self).__init__()
		self.W = nn.Parameter(torch.zeros([embsize, outsize]))
		self.nolinear = nolinear
		self.embsize = embsize
		self.outsize = outsize

	def forward(self, embeddings, adj):
		n = embeddings.size()[0]
		ret = torch.sparse.mm(adj, embeddings).matmul(self.W)
		if self.nolinear == "ReLU":
			ret = F.relu(ret)
		return ret

class GCN(nn.Module):

	def __init__(self, embsize, hidsize, nclass):
		super(GCN, self).__init__()
		self.layer1 = singleGCNLayer(embsize, hidsize, "ReLU")
		self.layer2 = singleGCNLayer(hidsize, nclass)

	def forward(self, embeddings, adj):
		n = embeddings.size()[0]
		embeddings = self.layer1(embeddings, adj)
		embeddings = self.layer2(embeddings, adj)
		embeddings = torch.softmax(embeddings, dim = 1)
		return embeddings
