import os
import pickle
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from data import load_seq
from common import CustomDataset, data_prepare
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc


def collate_fn(data):
    """
    collate returns data in type long and bool

    Arguments:
        data: a list of samples fetched from `CustomDataset`

    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (# patiens) of type torch.float
    """

    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)

    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            masks[i_patient][j_visit][:len(visit)] = True
            x[i_patient][j_visit][:len(visit)] = torch.tensor(visit).type(torch.long)
            rev_masks[i_patient][len(patient) - 1 - j_visit][:len(visit)] = True
            rev_x[i_patient][len(patient) - 1 - j_visit][:len(visit)] = torch.tensor(visit).type(torch.long)

    return x, masks, rev_x, rev_masks, y


# AlphaAttention generates one attention value for each visit.
class AlphaAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        """
        Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;

        Arguments:
            embedding_dim: the embedding dimension
        """

        self.a_att = nn.Linear(embedding_dim, 1)

    def forward(self, g, rev_masks):
        """

        Arguments:
            g: the output tensor from RNN-alpha of shape (batch_size, # visits, embedding_dim)
            rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            alpha: the corresponding attention weights of shape (batch_size, # visits, 1)
        """
        att_score = self.a_att(g).squeeze(-1)  # (batch, visit, embedding) -> (batch, visit, 1)
        visit_masks = torch.sum(rev_masks, dim=-1).type(torch.bool)
        att_score = att_score.masked_fill(~visit_masks, -1e9)
        att_score = torch.softmax(att_score, dim=1).unsqueeze(-1)
        return att_score


# BetaAttention generate attention weight for each embedding in each visit
class BetaAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        """
        Define the linear layer `self.b_att` for beta-attention using `nn.Linear()`;

        Arguments:
            embedding_dim: the embedding dimension
        """

        self.b_att = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, h):
        """

        Arguments:
            h: the output tensor from RNN-beta of shape (batch_size, # visits, embedding_dim)

        Outputs:
            beta: the corresponding attention weights of shape (batch_size, # visitsseq_length, embedding_dim)

        """

        return torch.tanh(self.b_att(h))


def attention_sum(alpha, beta, rev_v, rev_masks):
    """
    mask select the hidden states for true visits (not padding visits) and then sum the them up.

    Arguments:
        alpha: the alpha attention weights of shape (batch_size, # visits, 1)
        beta: the beta attention weights of shape (batch_size, # visits, embedding_dim)
        rev_v: the visit embeddings in reversed time of shape (batch_size, # visits, embedding_dim)
        rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        c: the context vector of shape (batch_size, embedding_dim)
    """

    visit_mask = torch.sum(rev_masks, dim=-1).type(torch.bool).unsqueeze(-1)
    alpha_filtered = alpha.masked_fill(~visit_mask, 0)
    prod = alpha_filtered * beta * rev_v
    c = torch.sum(prod, dim=1)
    return c


# sums all diagnosis code embeddings in the same visit to form a single embedding for each visit
def sum_embeddings_with_mask(x, masks):
    """
    Mask select the embeddings for true visits (not padding visits) and then sum the embeddings for each visit up.

    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    """

    x = x * masks.unsqueeze(-1)
    x = torch.sum(x, dim=-2)
    return x


class RETAIN(nn.Module):

    # num_codes is total number of possible diagnosis code.
    def __init__(self, num_codes, embedding_dim=128):
        super().__init__()
        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(embedding_dim)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(embedding_dim)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(embedding_dim, 1)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            rev_x: the diagnosis sequence in reversed time of shape (batch_size, # visits, # diagnosis codes)
            rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embegginds through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        alpha = self.att_a(g, rev_masks)
        beta = self.att_b(h)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze(dim=-1)


def eval(model, val_loader):
    """
    Evaluate the model.

    Arguments:
        model: the RNN model
        val_loader: validation dataloader

    Outputs:
        precision: overall precision score
        recall: overall recall score
        f1: overall f1 score
        roc_auc: overall roc_auc score

    REFERENCE: checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(x, masks, rev_x, rev_masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


retain = RETAIN(num_codes=492)  # total vocab 491

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(retain.parameters(), lr=1e-3)


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
            y_hat = model(x, masks, rev_x, rev_masks)
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
        pr_auc: precision-recall auc score
    """

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit =  model(x, masks, rev_x, rev_masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return p, r, f, roc_auc, pr_auc


def model_retain():
    train_loader, val_loader, test_loader = data_prepare(load_seq, 32, collate_fn)
    # test loader result
    loader_iter = iter(train_loader)
    x, masks, rev_x, rev_masks, y = next(loader_iter)

    n_epochs = 1
    print(time.strftime("%H:%M:%S", time.localtime()))
    train(retain, train_loader, val_loader, n_epochs)
    print(time.strftime("%H:%M:%S", time.localtime()))

    p, r, f, roc_auc, pr_auc = full_eval(retain, test_loader)
    print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))

    torch.save(retain.state_dict(), "models/retain_test.pth")
