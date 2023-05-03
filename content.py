import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from common import full_eval

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

num_codes = 492

doc_train = load_pkl("resource/X_train.pkl")
lb_train = load_pkl("resource/Y_train.pkl")
doc_val = load_pkl("resource/X_valid.pkl")
lb_val = load_pkl("resource/Y_valid.pkl")
doc_test = load_pkl("resource/X_test.pkl")
lb_test = load_pkl("resource/Y_test.pkl")


# Split complete sequence into sub-seq each associated with one clear label.
# Converts codes of each visit to multi hot vector
# TODO: issue: sequence length, step size?
# TODO: do we need to handle cases with consecutive `1`s?
def split_sequence_hot_code(docs, labels):
    split_sequences = []
    split_labels = []
    idx_to_patient = []
    for i in range(len(docs)):
        patient_seq = docs[i]
        patient_labels = labels[i]
        for j in range(len(patient_seq)):
            sub_seq = patient_seq[0:j+1]
            seq_hc = []
            for visit in sub_seq:
                visit_hc = [0] * (num_codes-1)
                for mcode in visit:
                    visit_hc[mcode-1] = 1
                seq_hc.append(visit_hc)
            split_sequences.append(seq_hc)
            split_labels.append(patient_labels[j])
            idx_to_patient.append(i)
    return split_sequences, split_labels


seq_train, labels_train = split_sequence_hot_code(doc_train, lb_train)
seq_val, labels_val = split_sequence_hot_code(doc_val, lb_val)
seq_test, labels_test = split_sequence_hot_code(doc_test, lb_test)

save_pkl('./resource/X_train_mhc.pkl', seq_train)
save_pkl('./resource/Y_train_mhc.pkl', labels_train)
save_pkl('./resource/X_valid_mhc.pkl', seq_val)
save_pkl('./resource/Y_valid_mhc.pkl', labels_val)
save_pkl('./resource/X_test_mhc.pkl', seq_test)
save_pkl('./resource/Y_test_mhc.pkl', labels_test)

seq_train = load_pkl('./resource/X_train_mhc.pkl')
labels_train = load_pkl('./resource/Y_train_mhc.pkl')
seq_val = load_pkl('./resource/X_valid_mhc.pkl')
labels_val = load_pkl('./resource/Y_valid_mhc.pkl')
seq_test = load_pkl('./resource/X_test_mhc.pkl')
labels_test = load_pkl('./resource/Y_test_mhc.pkl')



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



dataset_train = CustomDataset(seq_train, labels_train)
train_loader = DataLoader(dataset_train, batch_size=16, collate_fn=collate_fn, shuffle=True)

dataset_val = CustomDataset(seq_val, labels_val)
val_loader = DataLoader(dataset_val, batch_size=16, collate_fn=collate_fn, shuffle=False)

dataset_test = CustomDataset(seq_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate_fn, shuffle=False)


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
            x: the multi hot encoded visits (# visits, batch_size, # total diagnosis codes)
            masks: the padding masks of shape (# visits, batch_size, # total diagnosis codes)

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
        # TODO: could also sum all visit embeddings up
        # (batch, n_topic) / (batch, 1)
        mean_u = torch.sum(lu * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        mean_log_sigma = torch.sum(ls * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        # generate from learned distribution
        gen = torch.randn(mean_u.shape) * torch.exp(mean_log_sigma) + mean_u  # (batch, n_topic)
        return gen


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
            x: the multi hot encoded visits (# visits, batch_size, # total diagnosis codes)
            masks: the padding masks of shape (# visits, batch_size, # total diagnosis codes)
        """
        # x = x.type(dtype=torch.float)
        x_embed = self.fc_embedding(x)
        output, _ = self.rnn(x_embed)
        final_visit_h = get_last_visit(output, masks)  # (batch_size, hidden_dim)
        topics = self.recognition(x, masks)  # (batch_size, n_topic)
        score = self.fc_q(final_visit_h) + self.fc_b(topics)  # (batch_size, 1)
        return self.sigmoid(score).squeeze(dim=-1)


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


ctn = Content(input_dim=num_codes-1)  # total vocab 491

# load the loss function
criterion = nn.BCELoss()
# load the optimizer
optimizer = torch.optim.Adam(ctn.parameters(), lr=0.00002)


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


n_epochs = 5
train(ctn, train_loader, val_loader, n_epochs)

p, r, f, roc_auc = eval(ctn, test_loader)
print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}'.format(p, r, f, roc_auc))

torch.save(ctn.state_dict(), "models/content_statedict.pth")

# reload and evaluation
model = Content(input_dim=num_codes-1)
model.load_state_dict(torch.load("models/content_statedict.pth"))
p, r, f, roc_auc, pr_auc = full_eval(model, test_loader)
print('Test p: {:.4f}, r:{:.4f}, f: {:.4f}, roc_auc: {:.4f}, pr_auc: {:.4f}'.format(p, r, f, roc_auc, pr_auc))
