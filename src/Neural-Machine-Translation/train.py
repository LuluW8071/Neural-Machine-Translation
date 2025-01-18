import comet_ml
import os
import io
import argparse
import pytorch_lightning as pl
import torch

import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.font_manager import FontProperties
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics.text import SacreBLEUScore

import utils

from dataset import NMTDataModule
from model import NMTModel
from checkpoint import load_checkpoint

# Load API
from dotenv import load_dotenv
load_dotenv()


class NMTTrainer(pl.LightningModule):
    """ Neural Machine Translation Trainer """
    def __init__(self, model, data_module, args):
        super(NMTTrainer, self).__init__()
        self.model = model
        self.args = args

        self.input_lang = data_module.input_lang
        self.output_lang = data_module.output_lang

        # For Attention Mapping Plots
        self.custom_font = FontProperties(fname=args.font_path, size=10)
        self.english_font = FontProperties(size=10) 

        # Loss fn and Metrics
        self.val_losses = []
        self.quadgram_bleu_scores = []

        self.loss_fn = nn.CrossEntropyLoss()
        self.quadgram_sacbleu = SacreBLEUScore(n_gram=4, smooth=True, tokenize='13a')

        # Precompute sync_dist for distributed GPUs train
        self.sync_dist = True if args.gpus > 1 else False

    def forward(self, input_tensor, target_tensor):
        """ Forward pass of the model """
        return self.model(input_tensor, target_tensor)


    def configure_optimizers(self):
        """ Configure optimizer and scheduler """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.args.lr_factor,
                patience=self.args.lr_patience,
                threshold=self.args.min_lr_threshold,
                threshold_mode='rel'
            ),
            'monitor': 'val_loss',              # Metric to monitor
            'interval': 'epoch',                # Scheduler step every epoch
            'frequency': 1
        }
        return [optimizer], [scheduler]


    def _common_step(self, batch, batch_idx):
        """ Common step for training and validation """
        input_tensor, target_tensor = batch
        decoder_out, _, attn_weights = self.forward(input_tensor, target_tensor)

        loss = self.loss_fn(decoder_out.view(-1, decoder_out.size(-1)), target_tensor.view(-1))

        return loss, decoder_out, input_tensor, target_tensor, attn_weights

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._common_step(batch, batch_idx)

        # Log train_loss in the logger
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss

    def validation_step(self, batch, batch_idx):
        # NOTE: Adopting teacher forcing to account for variable length to prevent overfitting
        # In validation, teacher forcing should not be done as it should simulate real-world examples
        # Constructed a test_step to simulate without teacher forcing using the same dataloader as in validation_step 
        loss, decoder_out, input_tensor, target_tensor, attn_weights = self._common_step(batch, batch_idx)
        self.val_losses.append(loss)

        # Decode the outputs and targets for metric calculations
        decoded_sentences = self._decode(decoder_out)
        targets = [utils.sentenceFromIndexes(self.output_lang, tgt.tolist()) for tgt in target_tensor]
        
        # Log some examples
        if batch_idx % 256 == 0:
            self._logger(batch_idx, input_tensor, targets, decoded_sentences, attn_weights, phase="Validation")

        # Calculate metrics
        quadgram_bleu_batch = self.quadgram_sacbleu(decoded_sentences, [targets])
        self.quadgram_bleu_scores.append(quadgram_bleu_batch)

        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """ Log validation metrics after validation epoch end """
        avg_val_loss = torch.stack(self.val_losses).mean()
        avg_quad_bleu = torch.stack(self.quadgram_bleu_scores).mean()

        # Log avg. loss and metrics
        metrics = {
            "val_loss": avg_val_loss, 
            "quad_bleu": avg_quad_bleu
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        self.val_losses.clear()
        self.quadgram_bleu_scores.clear()

    def test_step(self, batch, batch_idx):
        """ Perform Validation on same validation set without teacher forcing """
        input_tensor, target_tensor = batch
        decoder_out, _, attn_weights = self.forward(input_tensor, None)

        # Decode outputs and targets
        decoded_sentences = self._decode(decoder_out)
        targets = [utils.sentenceFromIndexes(self.output_lang, tgt.tolist()) for tgt in target_tensor]

        if batch_idx % 32 == 0:
            self._logger(batch_idx, input_tensor, targets, decoded_sentences, attn_weights, phase="Test")

        # Calculate BLEU
        quadgram_bleu_score = self.quadgram_sacbleu(decoded_sentences, [targets])
        metrics = {"test_quad_bleu": quadgram_bleu_score}
        
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)
        return metrics


    def _decode(self, decoded_output):
        """ Turn raw outputs back to sentence """
        _, topi = decoded_output.topk(1)
        decoded_ids = topi.squeeze().tolist()

        batch_sentences = []    # List to hold sentences for the entire batch
        for ids in decoded_ids:
            decoded_words = []
            for idx in ids:
                if idx == 1:    # EOS token: 1
                    # decoded_words.append('<EOS>')
                    break
                decoded_words.append(self.output_lang.index2word[idx])
            sentence = ' '.join(decoded_words)
            batch_sentences.append(sentence)

        return batch_sentences
    
    def _logger(self, batch_idx, input_tensors, target_tensors, decoded_sentences, attn_weights, phase):
        """ Log source, target, and translations """
        log_texts = []
        for i in range(32):
            log_source = utils.sentenceFromIndexes(self.input_lang, input_tensors[i].tolist())
            log_target, log_translated = target_tensors[i], decoded_sentences[i]
            log_text = f"{log_source}\n> {log_target}\n= {log_translated}"
            log_texts.append(log_text)
            combined_logs = "\n\n".join(log_texts)

            # 20% chances of log images (to control excessive log of images)
            if self.args.attention and torch.rand(1)>0.8:   
                translated_words = log_translated.split(' ')
                buf = self._showAttention(log_source, translated_words, attn_weights[i][:len(translated_words)])
                
                # Convert the buffer to a PIL image
                img = Image.open(buf)
                metadata = {
                    "input_sentence": log_source,
                    "target": log_target,
                    "translated": log_translated
                }
                self.logger.experiment.log_image(img, metadata=metadata)

                buf.close()

        # Log source, target and translated texts
        self.logger.experiment.log_text(text=combined_logs, metadata={"Phase": phase})


    def _showAttention(self, input_sentence, output_words, attentions):
        """ Attention plot """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        input_words = input_sentence.split(' ')
        attentions = attentions[:len(output_words), :len(input_words)]
        cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        ax.set_xticklabels([''] + input_words, 
                           rotation=90, 
                           fontproperties=self.english_font if self.args.reverse else self.custom_font)
        ax.set_yticklabels([''] + output_words, 
                           fontproperties=self.custom_font if self.args.reverse else self.english_font)

        # Set the tick positions and show labels at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf



def main(args):
    # Set seed for reproducibility
    # utils.set_seed(args.seed)

    # Data Module
    data_module = NMTDataModule(
        train_path=args.train_path,
        valid_path=args.valid_path,
        lang1=args.input_lang,          # Source language
        lang2=args.output_lang,         # Target language
        batch_size=args.batch_size,     # Batch size
        max_len=args.max_len,           # Maximum sequence length
        min_len=args.min_len,           # Minimum sequence length
        num_workers=args.num_workers,   # DataLoader workers
        reverse=args.reverse            # Whether to reverse the language pairs
    )

    data_module.setup()

    # Save the vocabs
    data_module.input_lang.save_to_file("input_vocab.json", input=True)
    data_module.output_lang.save_to_file("output_vocab.json", input=False)

    # Initialize the model
    h_params = {
        "input_size": data_module.input_lang.n_words,
        "output_size": data_module.output_lang.n_words,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "max_len": args.max_len,
        "bidirection": args.bidirection,
        "dropout_rate": 0.2,
        "attention": args.attention,
        "model_type": args.model_type,
        "device": args.device
    }

    model = NMTModel(**h_params)
    
    if args.checkpoint_path:
        encoder_state_dict, decoder_state_dict = load_checkpoint(args.checkpoint_path)
        model.encoder.load_state_dict(encoder_state_dict)
        model.decoder.load_state_dict(decoder_state_dict) 

    # Initialize the trainer and logger    
    nmt_trainer = NMTTrainer(model, data_module, args)
    
    comet_logger = CometLogger(
        api_key=os.getenv('API_KEY'), 
        project_name=os.getenv('PROJECT_NAME')
    )

    # Checkpoint Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",       
        filename='nmt-{epoch:02d}-{val_loss:.3f}-{val_quad_bleu:.3f}',                                             
        save_top_k=2,
        mode='min',
        save_weights_only=True      # Save only the weights cuz checkpoint is too large
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'min_epochs': 1,
        'max_epochs': args.epochs,                              
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'gradient_clip_val': args.grad_clip,                            # Gradient norm clipping value
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=4),
                      checkpoint_callback],
        'logger': comet_logger                                          # Logger to use for training
    }

    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
        
    if args.acc_grad > 1:
        trainer_args['accumulate_grad_batches'] = args.acc_grad

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(nmt_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(nmt_trainer, data_module)
    trainer.test(nmt_trainer, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, help='device to use for training')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Dataset Configuration
    parser.add_argument('--train_path', default=None, required=True, type=str, help='tsv file to load training data')
    parser.add_argument('--valid_path', default=None, required=True, type=str, help='csv file to load valid data')
    parser.add_argument('--font_path', default=None, required=True, type=str, help='Font_file: .ttf may require if data contains devangari like fonts')
    parser.add_argument('--input_lang', default='np', type=str, help='source language')
    parser.add_argument('--output_lang', default='en', type=str, help='target language')
    parser.add_argument('--reverse', action='store_true', help='Whether to reverse source and target languages')
    parser.add_argument('--max_len', default=12, type=int, help='maximum sequence length')
    parser.add_argument('--min_len', default=2, type=int, help='minimum sequence length')


    # Model HyperParameters
    parser.add_argument('-mt', '--model_type', default='lstm', type=str, help='model type to choose between lstm and gru')
    parser.add_argument('-hs','--hidden_size', default=128, type=int, help='model hidden size')
    parser.add_argument('-nl', '--num_layers', default=2, type=int, help='number of layers')
    parser.add_argument('-bd', '--bidirection', action='store_true', help='whether to use bidirectional model')
    parser.add_argument('-at', '--attention', action='store_true', help='whether to use attention mechanism')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    
    parser.add_argument('-lr','--learning_rate', default=4e-4, type=float, help='learning rate')
    parser.add_argument('-lrf', '--lr_factor', default=0.5, type=float, help='learning rate factor for decay')
    parser.add_argument('-lrp', '--lr_patience', default=1, type=int, help='learning rate patience for decay')
    parser.add_argument('-mlt', '--min_lr_threshold', default=1e-2, type=float, help='minimum learning rate threshold')

    parser.add_argument('--precision', default='32-true', type=str, help='precision')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path of checkpoint file to resume training')
    parser.add_argument('-gc', '--grad_clip', default=1.0, type=float, help='gradient norm clipping value')
    parser.add_argument('-ag', '--acc_grad', default=2, type=int, help='number of batches to accumulate gradients over')

    args = parser.parse_args()
    main(args)