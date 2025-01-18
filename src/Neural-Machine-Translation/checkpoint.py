import torch
from collections import OrderedDict


def load_checkpoint(checkpoint_path):
    """Loads a checkpoint and separates encoder and decoder state dictionaries before loading them into the model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint['state_dict']

    # Initialize state dictionaries
    encoder_state_dict = OrderedDict()
    decoder_state_dict = OrderedDict()

    # Separate encoder and decoder state dictionaries
    for k, v in model_state_dict.items():
        if k.startswith('model.encoder.'):
            name = k.replace('model.encoder.', '')
            encoder_state_dict[name] = v
        elif k.startswith('model.decoder.'):
            name = k.replace('model.decoder.', '')
            decoder_state_dict[name] = v

    return encoder_state_dict, decoder_state_dict