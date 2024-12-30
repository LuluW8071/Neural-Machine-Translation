import comet_ml
import os
import argparse
import pytorch_lightning as pl
import torch

import torch.optim as optim
import torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics import BLEUScore

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import NMTDataset
from model import NMT
from utils import set_seed

class NMTTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(NMTTrainer, self).__init__()
        self.model = model
        self.args = args

        self.losses = []
        self.bleu_scores = []

        # Metrics
        self.bleu = BLEUScore(n_gram=4, smooth=False)

        # Ignore padding when computing loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # Precompute sync_dist for distributed GPUs train
        self.sync_dist = True if args.gpus > 1 else False

    
    def forward(self):
        pass


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.args.lr_factor,     # Reduce LR by multiplying it by 0.8
                patience=self.args.lr_patience, # No. of epochs to wait before reducing LR
                threshold=self.lr_threshold,    # Minimum change in val_loss to qualify as improvement
                threshold_mode='rel',           # Relative threshold (e.g., 0.1% change)
                min_lr=self.min_lr              # Minm. LR to stop reducing
            ),
            'monitor': 'val_loss',              # Metric to monitor
            'interval': 'epoch',                # Scheduler step every epoch
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def common_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, help='device to use for training')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Dataset Configuration
    parser.add_argument('--file_path', default=None, required=True, type=str, help='csv file to load training data')
    parser.add_argument('--input_lang', default='en', type=str, help='source language')
    parser.add_argument('--output_lang', default='nep', type=str, help='target language')
    parser.add_argument('--reverse', default=False, action='store_true', help='whether to reverse source and target languages')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    
    parser.add_argument('-lr','--learning_rate', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-lrf', '--lr_factor', default=0.5, type=float, help='learning rate factor for decay')
    parser.add_argument('-lrp', '--lr_patience', default=2, type=int, help='learning rate patience for decay')
    parser.add_argument('-mlt', '--min_lr_threshold', default=5e-3, type=float, help='minimum learning rate threshold')
    parser.add_argument('-mlr', '--min_lr', default=5e-6, type=float, help='minimum learning rate')

    parser.add_argument('--precision', default='32-true', type=str, help='precision')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path of checkpoint file to resume training')
    parser.add_argument('-gc', '--grad_clip', default=0.8, type=float, help='gradient norm clipping value')
    parser.add_argument('-ag', '--accumulate_grad', default=2, type=int, help='number of batches to accumulate gradients over')

    args = parser.parse_args()
    main(args)