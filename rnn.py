import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from data import load_mhc
from common import CustomDataset, collate_fn, train, full_eval, get_last_visit, data_prepare

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

num_codes = 492


class RNN(torch.nn.Module):
    """
    Simple GRU with embedding
    """

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


def model_rnn():
    train_loader, val_loader, test_loader = data_prepare(load_mhc, 16, collate_fn)

    rnn = RNN(input_dim=num_codes - 1)  # total vocab 491

    # load the loss function
    criterion = nn.BCELoss()
    # load the optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    n_epochs = 5
    print(time.strftime("%H:%M:%S", time.localtime()))
    train(rnn, train_loader, val_loader, n_epochs, criterion, optimizer)
    print(time.strftime("%H:%M:%S", time.localtime()))

    torch.save(rnn.state_dict(), "models/rnn_test.pth")

    p, r, f, roc_auc, pr_auc = full_eval(rnn, test_loader)
    print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))
