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

import utils
from dataset import NMTDataModule
from model import NMTModel


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

    
    def forward(self, input_tensor, target_tensor):
        return self.model(input_tensor, target_tensor, self.args.max_len)


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

    def _common_step(self, batch, batch_idx):
        # Unpack batch
        input_tensor, target_tensor = batch
        decoder_out, _ = self.forward(input_tensor, target_tensor)

        loss = self.loss_fn(decoder_out.view(-1, decoder_out.size(-1)), target_tensor.view(-1))
        return loss, decoder_out


    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx)

        # Log train_loss in the logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        input_tensor, target_tensor = batch
        loss, decoded_output = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Decode the outputs
        decoded_words = self._decode(decoded_output)
        source = utils.sentenceFromIndexes(self.input_lang, input_tensor.tolist())
        targets = utils.sentenceFromIndexes(self.output_lang, target_tensor.tolist())
        
        if batch_idx % 100 == 0:
            log_source, log_targets = source[-1], targets[-1]
            log_texts = f"Source: {log_source}\n> {log_targets}\n= {decoded_words[-1]}\n\n"
            self.logger.experiment.log_text(text = log_texts)
            
        # Calculate the metrics
        bleu_batch = self.bleu(decoded_words, targets)

        self.bleu_scores.append(bleu_batch)
        return {'val_loss': loss}    


    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        avg_bleu = torch.stack(self.bleu_scores).mean()

        # Log all metrics using log_dict
        metrics = {
            'val_loss': avg_loss,
            'val_bleu': avg_bleu
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear the lists for the next epoch
        self.losses.clear()
        self.bleu_scores.clear()
    
    def _decode(self, decoded_output):
        # Turn back to sentence
        _, topi = decoded_output.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == 1:     # EOS token: 1
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.output_lang.index2word[idx.item()])

        return decoded_words
    

def main(args):
    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # Data Module
    data_module = NMTDataModule(
        file_path=args.file_path,       # Path to your dataset file
        lang1=args.input_lang,          # Source language
        lang2=args.output_lang,         # Target language
        split_ratio=0.8,                # Train-test split
        batch_size=args.batch_size,     # Batch size
        max_len=args.max_len,           # Maximum sequence length
        num_workers=args.num_workers,   # DataLoader workers
        seed=args.seed,                 # Random seed
        reverse=args.reverse            # Whether to reverse the language pairs
    )

    data_module.setup()

    # Initialize the model
    h_params = {
        "input_size": data_module.input_lang.n_words,
        "output_size": data_module.output_lang.n_words,
        "hidden_size": 128,
        "num_layers": 1,
        "max_len": args.max_len,
        "dropout_rate": 0.1, 
    }

    model = NMTModel(**h_params)
    # model = torch.compile(model)

    nmt_trainer = NMTTrainer(model, args)

    # Initialize the trainer
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'), project_name=os.getenv('PROJECT_NAME'))

    # Checkpoint Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",       
        filename='nmt-{epoch:02d}-{val_bleu:.3f}',                                             
        save_top_k=2,
        mode='min'
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'min_epochs': 1,                                                # Minm. no. of epochs to run
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'gradient_clip_val': args.grad_clip,                            # Gradient norm clipping value
        'accumulate_grad_batches': args.accumulate_grad,                # No. of batches to accumulate gradients over
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=3),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }

    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(nmt_trainer, data_module)
    trainer.validate(nmt_trainer, data_module)

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
    parser.add_argument('--max_len', default=12, type=int, help='maximum sequence length')
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