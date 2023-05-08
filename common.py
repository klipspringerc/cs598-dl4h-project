import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *

from data import num_codes, window_size, load_mhc, load_seq, split_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc


class CustomDataset(Dataset):

    def __init__(self, docs, labels):
        self.x = docs
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def collate_fn(data):
    """
    collate returns data in type float and bool

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

    max_num_visits = max(num_visits)
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


def data_prepare(loader=load_mhc, batch_size=16, collate=collate_fn):
    seq_train, labels_train, seq_val, labels_val, seq_test, labels_test = loader()

    dataset_train = CustomDataset(seq_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate, shuffle=True)

    dataset_val = CustomDataset(seq_val, labels_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, collate_fn=collate, shuffle=False)

    dataset_test = CustomDataset(seq_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate, shuffle=False)
    return train_loader, val_loader, test_loader


def get_last_visit(hidden_states, masks):
    """
    Retrieve hidden states from the last valid visit based on masks

    Arguments:
        hidden_states: the hidden states of each visit of shape (batch_size, # visits, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim)

    First convert the mask to a vector of shape (batch_size,) containing the true visit length;
          and then use this length vector as index to select the last visit.
    """
    idx = torch.sum(torch.sum(masks, -1) > 0, -1)
    # pass two list in index [], so that each row would select different index according to idx.
    return hidden_states[range(hidden_states.shape[0]), idx - 1, :]


def train(model, train_loader, val_loader, n_epochs, criterion, optimizer):
    """
    Train the model.

    Arguments:
        model: the RNN model
        train_loader: training dataloder
        val_loader: validation dataloader
        n_epochs: total number of epochs
        criterion: loss function
        optimizer: optimizer
    """

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks)
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
        y_logit = model(x, masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


def full_eval(model, val_loader):
    """
    Evaluate the model. P,R,F1 assumes threshold of 0.5.

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
        y_logit = model(x, masks)
        y_hat = torch.where(y_logit > 0.5, 1, 0)
        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return p, r, f, roc_auc, pr_auc


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def get_test_mhc_loader(collate=collate_fn):
    seq_test = load_pkl('./resource/X_test_mhc.pkl')
    labels_test = load_pkl('./resource/Y_test_mhc.pkl')
    dataset_test = CustomDataset(seq_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate, shuffle=False)
    return test_loader


def get_test_seq_loader(collate=collate_fn):
    doc_test = load_pkl("resource/X_test.pkl")
    label_test = load_pkl("resource/Y_test.pkl")
    seq_test, labels_test = split_sequence(doc_test, label_test)
    dataset_test = CustomDataset(seq_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate, shuffle=False)
    return test_loader
