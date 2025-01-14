import torch.nn as nn

from models.gru_encoder import GRUEncoder
from models.lstm_encoder import LSTMEncoder

from models.gru_decoder import GRUDecoder
from models.attn_gru_decoder import AttnGRUDecoder
from models.attn_lstm_decoder import AttnLSTMDecoder


class NMTModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, max_len, bidirection=True, dropout_rate=0.1, attention=True, model_type="lstm", device="cpu"):
        super(NMTModel, self).__init__()
        self.encoder = (
            GRUEncoder(input_size, hidden_size, num_layers, bidirection, dropout_rate)
            if model_type == "gru"
            else 
            LSTMEncoder(input_size, hidden_size, num_layers, bidirection, dropout_rate)
        )

        self.decoder = (
            AttnGRUDecoder(hidden_size, output_size, num_layers, bidirection, device)
            if attention and model_type == "gru"
            else AttnLSTMDecoder(hidden_size, output_size, num_layers, bidirection, device)
            if attention and model_type == "lstm"
            else GRUDecoder(hidden_size, output_size, num_layers, device)
        )

        self.max_len = max_len

    def forward(self, input_tensor, target_tensor=None):
        # Encoder forward pass
        encoder_out, encoder_hidden = self.encoder(input_tensor)

        # Decoder forward pass
        decoder_outputs, decoder_hidden, attention = self.decoder(encoder_out, encoder_hidden, self.max_len, target_tensor)

        return decoder_outputs, decoder_hidden, attention