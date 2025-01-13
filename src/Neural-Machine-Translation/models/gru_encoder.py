""" Bi-GRU Encoder """

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirection=True, dropout_rate=0.1):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirection
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          num_layers,
                          bidirectional=bidirection,
                          batch_first=True,
                          dropout=dropout_rate)
        
        self.dropout = nn.Dropout(dropout_rate)

        # For Bidirectional
        self.fc = nn.Linear(self.hidden_size*2, self.hidden_size) if self.bidirectional else None

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        out, hidden = self.gru(embedded)
        
        if self.bidirectional:
            # Combine forward and backward hidden states across all layers
            hidden = torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:]), dim=2)
            hidden = self.fc(hidden)

        return out, hidden