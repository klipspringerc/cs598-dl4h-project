import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from common import CustomDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

num_codes = 492

seq_train = load_pkl('./resource/X_train_mhc.pkl')
labels_train = load_pkl('./resource/Y_train_mhc.pkl')
seq_val = load_pkl('./resource/X_valid_mhc.pkl')
labels_val = load_pkl('./resource/Y_valid_mhc.pkl')
seq_test = load_pkl('./resource/X_test_mhc.pkl')
labels_test = load_pkl('./resource/Y_test_mhc.pkl')


window_size = 64


def collate_fn(data):
    """

    Arguments:
        data: a list of samples fetched from `CustomDataset`

    Outputs:
        x: a tensor of shape (# patiens, max # visits, largest diagnosis code) of type torch.float, multi-host encoding of diagnosis code within each visit
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time.
        rev_masks: same as mask but in reversed time.
        y: a tensor of shape (# patiens) of type torch.float
    """

    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)

    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(max(num_visits), window_size)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.float)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.float)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            masks[i_patient][j_visit][:len(visit)] = True
            x[i_patient][j_visit][:len(visit)] = torch.tensor(visit).type(torch.float)
            rev_masks[i_patient][len(patient) - 1 - j_visit][:len(visit)] = True
            rev_x[i_patient][len(patient) - 1 - j_visit][:len(visit)] = torch.tensor(visit).type(torch.float)

    return x, masks, rev_x, rev_masks, y


dataset_train = CustomDataset(seq_train, labels_train)
train_loader = DataLoader(dataset_train, batch_size=16, collate_fn=collate_fn, shuffle=True)

dataset_val = CustomDataset(seq_val, labels_val)
val_loader = DataLoader(dataset_val, batch_size=16, collate_fn=collate_fn, shuffle=False)

dataset_test = CustomDataset(seq_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate_fn, shuffle=False)


class SimpleAttnR2(nn.Module):
    """
        Define the attention network with no RNN modules
        Flatten scoring over fixed window
    """

    def __init__(self, input_dim=num_codes-1, embedding_dim=100, key_dim=64, window_size=window_size, hidden_dim=512):
        super().__init__()
        self.window_size = window_size
        self.fc_key = nn.Linear(input_dim, key_dim)
        self.fc_score_1 = nn.Linear(window_size * key_dim, hidden_dim)
        self.fc_score_2 = nn.Linear(hidden_dim, window_size)
        self.fc_embed = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        """
        Arguments:
            x: the multi hot encoded visits in reverse order (batch_size, # visits, # total diagnosis codes)
            masks: the padding masks of shape in reverse order (batch_size, # visits, # total diagnosis codes)
        """
        # generate key, query and embeddings
        if x.shape[1] > self.window_size:
            x = x[:, :self.window_size, :]
            masks = masks[:, :self.window_size, :]
        # mlp to generate embedding weights
        x_key = self.fc_key(x)  # (batch, window_size, key_dim)
        x_key_flatten = x_key.flatten(start_dim=-2)  # (batch, window_size * key_dim)
        x_score = torch.relu(self.fc_score_1(x_key_flatten))  # (batch, hidden_dim)
        x_score = torch.relu(self.fc_score_2(x_score))  # (batch, window_size)
        x_embed = self.fc_embed(x)  # (batch, window_size, embed)
        visit_mask = torch.sum(masks, -1).type(torch.bool)  # (batch, window_size)
        # filter score by visit masks, then softmax along visit axis
        scores = x_score.masked_fill(visit_mask == 0, -1e9).softmax(dim=-1)  # (batch, window_size)
        output = torch.matmul(x_embed.transpose(-2, -1), scores.unsqueeze(-1)).squeeze(
            -1)  # (batch, embed, window_size) X (batch, window_size, 1) -> (batch, embed)
        return self.sigmoid(self.fc(output)).squeeze(dim=-1)


class SimpleAttn(nn.Module):
    """
        Define the attention network with no RNN modules
        Positional encoding, max window size in reverse order, use mean query and key vectors for embedding scoring
    """

    def __init__(self, input_dim=num_codes - 1, embedding_dim=128, key_dim=128, window_size=64, pos_enc_dim=8):
        super().__init__()
        self.window_size = window_size
        self.position_encoding = torch.tensor(get_position_encoding(350, pos_enc_dim)).type(torch.float).unsqueeze(0)
        self.fc_key = nn.Linear(input_dim + pos_enc_dim, key_dim)
        self.fc_query = nn.Linear(input_dim + pos_enc_dim, key_dim)
        # self.fc_score = nn.Linear(key_dim, 1)
        self.fc_embed = nn.Linear(in_features=input_dim + pos_enc_dim, out_features=embedding_dim)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        """
        Arguments:
            x: the multi hot encoded visits in reverse order (batch_size, # visits, # total diagnosis codes)
            masks: the padding masks of shape in reverse order (batch_size, # visits, # total diagnosis codes)
        """
        # generate key, query and embeddings
        if x.shape[1] > self.window_size:
            x = x[:, :self.window_size, :]
            masks = masks[:, :self.window_size, :]
        pos = self.position_encoding
        if pos.shape[1] > x.shape[1]:
            pos = pos[:, :x.shape[1], :]
        x = torch.cat((x, pos.expand(x.shape[0], x.shape[1], pos.shape[2])), -1)  # concat input with position encodings
        x_key = torch.relu(self.fc_key(x))  # (batch, #visit, key_dim)
        x_query = torch.relu(self.fc_query(x))  # (batch, #visit, key_dim)
        x_embed = self.fc_embed(x)  # (batch, #visit, embed)
        # use key and query similarity to generate score vector
        visit_mask = torch.sum(masks, -1).type(torch.bool)  # (batch, # visits)
        mean_query = torch.sum(x_query * visit_mask.unsqueeze(-1), dim=1) / torch.sum(visit_mask, dim=-1).unsqueeze(-1)  # (batch, key_dim)
        x_score = torch.matmul(x_key.unsqueeze(-1).transpose(-2,-1), mean_query.unsqueeze(-2).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # calc inner product between key and v
        # filter score by visit masks, then softmax along visit axis
        scores = x_score.masked_fill(visit_mask == 0, -1e9).softmax(dim=-1)  # (batch, #visit)
        output = torch.matmul(x_embed.transpose(-2, -1), scores.unsqueeze(-1)).squeeze(
            -1)  # (batch, embed, #visit) X (batch, #visit, 1) -> (batch, embed)
        return self.sigmoid(self.fc(output)).squeeze(dim=-1)


class SimpleAttnR1(nn.Module):
    """
        Define the attention network with no RNN modules
    """

    def __init__(self, input_dim=num_codes - 1, embedding_dim=100, key_dim=200):
        super().__init__()
        self.fc_key = nn.Linear(input_dim, key_dim)
        self.fc_query = nn.Linear(input_dim, key_dim)
        # self.fc_score = nn.Linear(key_dim, 1)
        self.fc_embed = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        """
        Arguments:
            x: the multi hot encoded visits (batch_size, # visits, # total diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # total diagnosis codes)
        """
        # generate key, query and embeddings
        x_key = self.fc_key(x)  # (batch, #visit, key_dim)
        x_query = self.fc_query(x)  # (batch, #visit, key_dim)
        x_embed = self.fc_embed(x)  # (batch, #visit, embed)
        # use key and query similarity to generate score vector
        visit_mask = torch.sum(masks, -1).type(torch.bool)  # (batch, # visits)
        mean_query = torch.sum(x_query * visit_mask.unsqueeze(-1), dim=1) / torch.sum(visit_mask, dim=-1).unsqueeze(-1)  # (batch, key_dim)
        x_score = torch.matmul(x_key.unsqueeze(-1).transpose(-2,-1), mean_query.unsqueeze(-2).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # calc inner product between key and v
        # x_score = self.fc_score(x_key).squeeze(-1)  # (batch, #visit)
        # filter score by visit masks, then softmax along visit axis
        # visit_mask = torch.sum(masks, -1) > 0  # (batch, #visit)
        scores = x_score.masked_fill(visit_mask == 0, -1e9).softmax(dim=-1)  # (batch, #visit)
        output = torch.matmul(x_embed.transpose(-2, -1), scores.unsqueeze(-1)).squeeze(
            -1)  # (batch, embed, #visit) X (batch, #visit, 1) -> (batch, embed)
        return self.sigmoid(self.fc(output)).squeeze(dim=-1)


position_encoding = torch.tensor(get_position_encoding(350, 8)).unsqueeze(0)

attn = SimpleAttn(input_dim=num_codes-1)

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(attn.parameters(), lr=0.001)


def eval(model, val_loader):
    """
    Evaluate the model.

    Arguments:
        model: the model
        val_loader: validation dataloader

    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(rev_x, rev_masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


def full_eval(model, val_loader):
    """
    Evaluate the model.

    Arguments:
        model: the model
        val_loader: validation dataloader

    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(rev_x, rev_masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return p, r, f, roc_auc, pr_auc

def train(model, train_loader, val_loader, n_epochs):
    """
    Train the model.

    Arguments:
        model: the RNN model
        train_loader: training dataloder
        val_loader: validation dataloader
        n_epochs: total number of epochs
    """

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(rev_x, rev_masks)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        p, r, f, roc_auc = eval(model, val_loader)
        print('Epoch: {} \t Validation p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}'.format(epoch + 1, p, r, f,
                                                                                               roc_auc))
    return round(roc_auc, 2)


n_epochs = 2
print(time.strftime("%H:%M:%S", time.localtime()))
train(attn, train_loader, val_loader, n_epochs)
print(time.strftime("%H:%M:%S", time.localtime()))

torch.save(attn.state_dict(), "models/simple_attn_epoch2.pth")
