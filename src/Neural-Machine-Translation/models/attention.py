import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, bidirection):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)       # Query 
        
        self.Ua = (nn.Linear(hidden_size * 2, hidden_size)
              if bidirection else
               nn.Linear(hidden_size, hidden_size))
        
        self.Va = nn.Linear(hidden_size, 1)                 # Values


    def forward(self, query, keys):
        """ Forward pass of Bahdanau Attention """
        # Query: Decoder Hidden State (bs, hidden_size)
        # Keys: Encoder Hidden State (bs, seq_len, hidden_size)

        # Take the last layer of the decoder hidden state and unsqueeze
        query = query[:, -1, :].unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Compute Attn scores by adding transformed query(Wa) and keys(Ua), follwed by tanh and finally transform the result via value layer(Va)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))

        # Adjust dimenstions for softmax and batch matrix multiplication by reshaping to (bs, 1, seq_len)
        scores = scores.squeeze(-1).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)

        # Compute context vector as weighted sum of keys
        # torch.bmm -> batch matrix multiplication
        context = torch.bmm(weights, keys)

        # Return context and weights
        return context, weights
