import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.attn_decoder import AttnDecoder


class NMTModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, max_len, bidirection=True, dropout_rate=0.1, attention=True, device="cpu"):
        super(NMTModel, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, bidirection, dropout_rate)
        self.decoder = (
            AttnDecoder(hidden_size, output_size, num_layers, device)
            if attention
            else Decoder(hidden_size, output_size, num_layers, device)
        )
        self.max_len = max_len

    def forward(self, input_tensor, target_tensor=None):
        # Encoder forward pass
        encoder_out, encoder_hidden = self.encoder(input_tensor)

        # Decoder forward pass
        decoder_outputs, decoder_hidden, attention = self.decoder(encoder_out, encoder_hidden, self.max_len, target_tensor)

        return decoder_outputs, decoder_hidden, attention