import os
import pickle
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from common import CustomDataset, train, eval, full_eval, data_prepare, get_last_visit, collate_fn
from data import convert_mhc_data, load_mhc

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

num_codes = 492


class Recognition(torch.nn.Module):

    def __init__(self, input_dim=num_codes-1, hidden_dim=200, topic_dim=50):
        super().__init__()
        """
        Define the recognition MLP that generates topic vector theta;

        Arguments:
            input_dim: generator does not take embeddings, directly put input dimension here
        """

        self.a_att = nn.Linear(input_dim, hidden_dim)
        self.b_att = nn.Linear(hidden_dim, hidden_dim)
        self.u_ln = nn.Linear(hidden_dim, topic_dim)
        self.sigma_ln = nn.Linear(hidden_dim, topic_dim)

        self.hidden = hidden_dim

    def forward(self, x, masks):
        """

        Arguments:
            x: the multi hot encoded visits (batch_size, # visits, # total diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # total diagnosis codes)

        Outputs:
            gen: generated value from learned distribution
        """
        # MLP to obtain mean and log_sigma values
        x = torch.relu(self.a_att(x))  # (batch, visit, input) -> (batch, visit, hidden)
        x = torch.relu(self.b_att(x))
        lu = self.u_ln(x)  # -> (batch, visit, n_topic)
        ls = self.sigma_ln(x)  # -> (batch, visit, n_topic)
        visit_masks = torch.sum(masks, dim=-1).type(torch.bool)  # (batch, visit)
        # calculate mean with mask
        # (batch, n_topic) / (batch, 1)
        mean_u = torch.sum(lu * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        mean_log_sigma = torch.sum(ls * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        # generate from learned distribution
        gen = torch.randn(mean_u.shape) * torch.exp(mean_log_sigma) + mean_u  # (batch, n_topic)
        return gen




class Content(torch.nn.Module):
    """
    Define the CONTENT network that contains recognition and GRU modules;
    """
    def __init__(self, input_dim=num_codes-1, embedding_dim=100, hidden_dim=200, topic_dim=50):
        """
        Arguments:
            input_dim: generator does not take embeddings, directly put input dimension here
        """
        super().__init__()

        self.fc_embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.recognition = Recognition(input_dim=input_dim, hidden_dim=hidden_dim, topic_dim=topic_dim)
        self.fc_q = nn.Linear(hidden_dim, 1)
        self.fc_b = nn.Linear(topic_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        """
        Arguments:
            x: the multi hot encoded visits (batch_size, # visits, # total diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # total diagnosis codes)
        """
        # x = x.type(dtype=torch.float)
        x_embed = self.fc_embedding(x)
        output, _ = self.rnn(x_embed)
        final_visit_h = get_last_visit(output, masks)  # (batch_size, hidden_dim)
        topics = self.recognition(x, masks)  # (batch_size, n_topic)
        score = self.fc_q(final_visit_h) + self.fc_b(topics)  # (batch_size, 1)
        return self.sigmoid(score).squeeze(dim=-1)


def model_content():
    train_loader, val_loader, test_loader = data_prepare(load_mhc, 16, collate_fn)

    ctn = Content(input_dim=num_codes - 1)  # total vocab 491

    # load the loss function
    criterion = nn.BCELoss()
    # load the optimizer
    optimizer = torch.optim.Adam(ctn.parameters(), lr=0.00002)

    n_epochs = 4
    print(time.strftime("%H:%M:%S", time.localtime()))
    train(ctn, train_loader, val_loader, n_epochs, criterion, optimizer)
    print(time.strftime("%H:%M:%S", time.localtime()))

    torch.save(ctn.state_dict(), "models/content_test.pth")

    # reload and evaluation
    model = Content(input_dim=num_codes - 1)
    model.load_state_dict(torch.load("models/content_opt.pth"))
    p, r, f, roc_auc, pr_auc = full_eval(model, test_loader)
    print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))
