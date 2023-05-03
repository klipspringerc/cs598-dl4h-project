import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from common import CustomDataset, collate_fn
from torch.utils.data import Dataset, DataLoader

num_codes = 492

seq_train = load_pkl('./resource/X_train_mhc.pkl')
labels_train = load_pkl('./resource/Y_train_mhc.pkl')
seq_val = load_pkl('./resource/X_valid_mhc.pkl')
labels_val = load_pkl('./resource/Y_valid_mhc.pkl')
seq_test = load_pkl('./resource/X_test_mhc.pkl')
labels_test = load_pkl('./resource/Y_test_mhc.pkl')

dataset_train = CustomDataset(seq_train, labels_train)
train_loader = DataLoader(dataset_train, batch_size=16, collate_fn=collate_fn, shuffle=True)

dataset_val = CustomDataset(seq_val, labels_val)
val_loader = DataLoader(dataset_val, batch_size=16, collate_fn=collate_fn, shuffle=False)

dataset_test = CustomDataset(seq_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate_fn, shuffle=False)


class SimpleAttn(nn.Module):
    """
        Define the attention network with no RNN modules
    """

    def __init__(self, input_dim=num_codes - 1, embedding_dim=100, key_dim=200):
        super().__init__()
        self.fc_key = nn.Linear(input_dim, key_dim)
        self.fc_score = nn.Linear(key_dim, 1)
        self.fc_embed = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        """
        Arguments:
            x: the multi hot encoded visits (# visits, batch_size, # total diagnosis codes)
            masks: the padding masks of shape (# visits, batch_size, # total diagnosis codes)
        """
        # generate score vector and embeddings
        x_key = self.fc_key(x)  # (batch, #visit, key_dim)
        x_score = self.fc_score(x_key).squeeze(-1)  # (batch, #visit)
        x_embed = self.fc_embed(x)  # (batch, #visit, embed)
        # filter score by visit masks, then softmax along visit axis
        visit_mask = torch.sum(masks, -1) > 0  # (batch, #visit)
        scores = x_score.masked_fill(visit_mask == 0, -1e9).softmax(dim=-1)  # (batch, #visit)
        output = torch.matmul(x_embed.transpose(-2, -1), scores.unsqueeze(-1)).squeeze(
            -1)  # (batch, embed, #visit) X (batch, #visit, 1) -> (batch, embed)
        return self.sigmoid(self.fc(output)).squeeze(dim=-1)


attn = SimpleAttn(input_dim=num_codes-1)

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(attn.parameters(), lr=0.001)

n_epochs = 2
print(time.strftime("%H:%M:%S", time.localtime()))
train(attn, train_loader, val_loader, n_epochs)
print(time.strftime("%H:%M:%S", time.localtime()))
