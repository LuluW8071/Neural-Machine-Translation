""" GRU Decoder """

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device):
        super(GRUDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        # NOTE: In NMT Bidirectional RNN Decoder doesn't make sense
        # as decoding is autoregressive process; each token is generated step-by-step, conidtioned on prev. tokens and encoder outputs.
        self.gru = nn.GRU(hidden_size, 
                          hidden_size,
                          num_layers, 
                          bidirectional=False, 
                          batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoder_out, encoder_hidden, max_len, target_tensor=None):
        # Get batch size of encoder output
        bs = encoder_out.size(0)

        # Initial decoder input (SOS token: 0)
        decoder_input = torch.empty(bs, 1, dtype=torch.long).fill_(0).to(self.device)  

        # Match decoder hidden state to encoder hidden state
        decoder_hidden = encoder_hidden

        # List to hold decoder outputs
        decoder_outputs = []
        for i in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None and torch.rand(1) > 0.2: 
                # Teacher forcing: Feed the target as the next input(80% chance)
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, decoder_input, decoder_hidden):
        output = self.embedding(decoder_input)
        output = F.relu(output)
        output, hidden = self.gru(output, decoder_hidden)
        output = self.out(output)
        return output, hidden