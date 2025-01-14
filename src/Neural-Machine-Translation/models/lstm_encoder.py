""" Bi-LSTM Encoder """

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirection=True, dropout_rate=0.1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirection
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, 
                            hidden_size, 
                            num_layers, 
                            bidirectional=bidirection, 
                            batch_first=True, 
                            dropout=dropout_rate)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)  # For Bidirectional

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        out, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            # Combine forward and backward hidden states across all layers
            hidden = torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:]), dim=2)
            cell = torch.cat((cell[0:self.num_layers], cell[self.num_layers:]), dim=2)
            
            # Apply a linear layer to reduce dimensions if needed
            hidden = self.fc(hidden)
            cell = self.fc(cell)

        return out, (hidden, cell)
