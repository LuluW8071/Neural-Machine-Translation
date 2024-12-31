import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Used Bidiretional to capture the full context of the input sequence, enriching token representations by considering both past and future information
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          bidirectional=False,
                        #   batch_first=True,
                          dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        out, hidden = self.gru(embedded)

        # Combine forward and backward hidden states across all layers
        # hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)
        # hidden = self.fc(torch.cat((hidden[0:1], hidden[1:2]), dim=1))
        print("Encoder Hidden", hidden.shape)
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        # In NMT Bidirectional RNN Decoder doesn't make sense
        # as decoding is autoregressive process; each token is generated step-by-step, donidtioned on prev. tokens and encoder outputs.
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers, bidirectional=False, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_out, encoder_hidden, max_len, target_tensor=None):
        # Get batch size of encoder output
        bs = encoder_out.size(1)
        # print("batch:", bs)

        # Initial decoder input (SOS token: 0)
        decoder_input = torch.empty(bs, 1, dtype=torch.long).fill_(0)   

        decoder_hidden = encoder_hidden

        decoder_outputs = []

        # print(decoder_input.shape, decoder_hidden.shape)

        for i in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, decoder_input, decoder_hidden):
        output = self.embedding(decoder_input)
        # print("Decoder Embeding Output:", output.shape) 
        output = F.relu(output)
        output, hidden = self.gru(output, decoder_hidden)
        output = self.out(output)
        return output, hidden