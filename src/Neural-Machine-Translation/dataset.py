import pytorch_lightning as pl
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import utils
from logger import logger
import os

class PrepData(Dataset):
    def __init__(self, file_path, lang1, lang2, reverse=False):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)
            lines = list(reader)

        self.pairs = [[utils.normalize_String(s) for s in line] for line in lines]
        if reverse:
            logger.info("Reversing targets")
            self.pairs = [list(reversed(p)) for p in self.pairs]

        self.input_lang = None
        self.output_lang = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError(f"Index {idx} out of range.")
        return self.pairs[idx]

    def get_pairs(self):
        return self.pairs


class NMTDataModule(pl.LightningDataModule):
    def __init__(self, train_path, valid_path, lang1, lang2, batch_size=32, num_workers=2, max_len=64, min_len=4, reverse=False):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.lang1 = lang1
        self.lang2 = lang2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.min_len = min_len
        self.reverse = reverse
        self.input_lang = None
        self.output_lang = None

    def _init_tokenizers(self, train_pairs):
        combined_text_1 = [pair[0] for pair in train_pairs]
        combined_text_2 = [pair[1] for pair in train_pairs]

        tmp_input_path = "input_corpus.txt"
        tmp_output_path = "output_corpus.txt"
        with open(tmp_input_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_text_1))
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_text_2))

        self.input_lang = utils.WordVocabulary(self.lang2 if self.reverse else self.lang1, input_text_path=tmp_input_path)
        self.output_lang = utils.WordVocabulary(self.lang1 if self.reverse else self.lang2, input_text_path=tmp_output_path)

        os.remove(tmp_input_path)
        os.remove(tmp_output_path)

    def _prepare_data_pairs(self, pairs, input_lang, output_lang, max_len, EOS_token):
        num_pairs = len(pairs)
        input_ids = np.zeros((num_pairs, max_len), dtype=np.int32)
        target_ids = np.zeros((num_pairs, max_len), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = utils.indexesFromSentence(input_lang, inp)[:max_len - 1]
            tgt_ids = utils.indexesFromSentence(output_lang, tgt)[:max_len - 1]
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        return input_ids, target_ids

    def setup(self, stage=None):
        train_data = PrepData(self.train_path, self.lang1, self.lang2, self.reverse)
        valid_data = PrepData(self.valid_path, self.lang1, self.lang2, self.reverse)

        train_pairs = train_data.get_pairs()
        valid_pairs = valid_data.get_pairs()

        self._init_tokenizers(train_pairs + valid_pairs)

        train_pairs = utils.filterPairs(train_pairs, self.max_len, self.min_len, tokenizer=self.input_lang)
        valid_pairs = utils.filterPairs(valid_pairs, self.max_len, self.min_len, tokenizer=self.input_lang)

        EOS_token = utils.EOS_Token

        input_ids_train, target_ids_train = self._prepare_data_pairs(train_pairs, self.input_lang, self.output_lang, self.max_len, EOS_token)
        input_ids_valid, target_ids_valid = self._prepare_data_pairs(valid_pairs, self.input_lang, self.output_lang, self.max_len, EOS_token)

        self.train_dataset = TensorDataset(torch.LongTensor(input_ids_train), torch.LongTensor(target_ids_train))
        self.valid_dataset = TensorDataset(torch.LongTensor(input_ids_valid), torch.LongTensor(target_ids_valid))

        self.input_lang.save_to_file("input_vocab.json")
        self.output_lang.save_to_file("output_vocab.json")

        logger.info(f"Input vocab size: {self.input_lang.n_words}")
        logger.info(f"Output vocab size: {self.output_lang.n_words}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
