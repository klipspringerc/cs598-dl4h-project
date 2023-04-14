import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


num_codes = 492


class Recognition(torch.nn.Module):

    def __init__(self, input_dim=num_codes, hidden_dim=200, topic_dim=50):
        super().__init__()
        """
        Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;

        Arguments:
            input_dim: generator does not take embeddings, directly put input dimension here
        """

        self.a_att = nn.Linear(input_dim, hidden_dim)
        self.b_att = nn.Linear(hidden_dim, hidden_dim)
        self.u_ln = nn.Linear(hidden_dim, topic_dim)
        self.sigma_ln = nn.Linear(hidden_dim, topic_dim)

        self.hidden = hidden_dim

    def forward(self, x, rev_masks):
        """

        Arguments:
            g: the output tensor from RNN-alpha of shape (batch_size, # visits, embedding_dim)
            rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            alpha: the corresponding attention weights of shape (batch_size, # visits, 1)

        HINT:
            1. Calculate the attention score using `self.a_att`
            2. Mask out the padded visits in the attention score with -1e9.
            3. Perform softmax on the attention score to get the attention value.
        """
        # your code here
        x = torch.relu(self.a_att(x))  # (batch, visit, input) -> (batch, visit, hidden)
        x = torch.relu(self.b_att(x))
        lu = self.u_ln(x)  # -> (batch, visit, n_topic)
        ls = self.sigma_ln(x)  # -> (batch, visit, n_topic)
        visit_masks = torch.sum(rev_masks, dim=-1).type(torch.bool)  # (batch, visit)
        # calculate mean with mask
        # (batch, n_topic) / (batch, 1)
        mean_u = torch.sum(lu * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        mean_log_sigma = torch.sum(ls * visit_masks.unsqueeze(-1), dim=1) / torch.sum(visit_masks, dim=-1).unsqueeze(-1)
        # generate from learned distribution
        gen = torch.randn(mean_u.shape) * torch.exp(mean_log_sigma) + mean_u
        return gen