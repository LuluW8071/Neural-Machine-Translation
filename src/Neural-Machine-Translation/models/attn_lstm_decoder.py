""" LSTM Decoder with Attention """

import torch
import torch.nn as nn

from .attention import BahdanauAttention

class AttnLSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, bidirection, device):
        super(AttnLSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        # NOTE: LSTM Decoder doesn't support bidirection for autoregressive generation
        self.lstm = nn.LSTM(3 * hidden_size if bidirection else 2 * hidden_size,
                            hidden_size, 
                            num_layers, 
                            bidirectional=False, 
                            batch_first=True)
        
        self.attention = BahdanauAttention(hidden_size, bidirection)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoder_out, encoder_hidden, max_len, target_tensor=None):
        # Get batch size of encoder output
        bs = encoder_out.size(0)

        # Initial decoder input (SOS token: 0)
        decoder_input = torch.empty(bs, 1, dtype=torch.long).fill_(0).to(self.device)  

        # Match decoder hidden and cell states to encoder hidden and cell states
        decoder_hidden, decoder_cell = encoder_hidden[0], encoder_hidden[1]

        # List to hold decoder outputs and attention scores
        decoder_outputs = []
        attentions = []

        for i in range(max_len):
            decoder_output, decoder_hidden, decoder_cell, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, decoder_cell, encoder_out)
            
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and torch.rand(1) > 0.2: 
                # Teacher forcing: Feed the target as the next input (80% chance)
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, (decoder_hidden, decoder_cell), attentions

    def forward_step(self, decoder_input, decoder_hidden, decoder_cell, encoder_out):
        input_embed = self.embedding(decoder_input)

        query = decoder_hidden.permute(1, 0, 2)
        
        context, attn_weights = self.attention(query, encoder_out)
        input_lstm = torch.cat((input_embed, context), dim=2)

        output, (hidden, cell) = self.lstm(input_lstm, (decoder_hidden, decoder_cell))
        output = self.out(output)

        return output, hidden, cell, attn_weights
