import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from common import CustomDataset, collate_fn, full_eval

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

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


def get_last_visit(hidden_states, masks):
    """
    Arguments:
        hidden_states: the hidden states of each visit of shape (batch_size, # visits, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim)

    First convert the mask to a vector of shape (batch_size,) containing the true visit length;
          and then use this length vector as index to select the last visit.
    """

    # your code here
    idx = torch.sum(torch.sum(masks, -1) > 0, -1)
    # pass two list in index [], so that each row would select different index according to idx.
    # note this is the way of index selecting
    return hidden_states[range(hidden_states.shape[0]), idx - 1, :]


class RNN(torch.nn.Module):

    def __init__(self, input_dim=num_codes-1, embedding_dim=100, hidden_dim=200):
        super().__init__()
        self.fc_embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        x_embed = self.fc_embedding(x)
        output, _ = self.rnn(x_embed)
        final_visit_h = get_last_visit(output, masks)  # (batch_size, hidden_dim)
        score = self.fc(final_visit_h)  # (batch_size, 1)
        return self.sigmoid(score).squeeze(dim=-1)


rnn = RNN(input_dim=num_codes-1)  # total vocab 491

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

n_epochs = 5
train(rnn, train_loader, val_loader, n_epochs)
print(time.strftime("%H:%M:%S", time.localtime()))

p, r, f, roc_auc, pr_auc = full_eval(rnn, test_loader)
print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))


