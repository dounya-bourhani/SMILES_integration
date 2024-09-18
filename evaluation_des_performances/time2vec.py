import torch
from torch import nn


class Time2Vec(nn.Module):
    """
    Transforms time into a vector
    This vector is used as a positional encoding for the transformer_aggregator

    Please see https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, time_dim):
        super(Time2Vec, self).__init__()

        self.time_dim = time_dim
        self.linear = nn.Linear(1, self.time_dim)

    def forward(self, t):
        """
        Computes the positional encoding for a tensor of times
        For a given time, its corresponding positional encoding
        is a vector of size time_dim.

        Args:
            t (tensor of N x 1 times): times list

        Output:
            time encoding (tensor of size N x time_dim)
        """
        x = self.linear(t)
        return torch.cat((x[:1], torch.sin(x[1:])))
