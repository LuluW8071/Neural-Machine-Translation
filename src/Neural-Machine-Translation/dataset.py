import pytorch_lightning as pl
import csv
import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    RandomSampler,
    SequentialSampler
)

import utils
from logger import logger

class PrepData(Dataset):
    def __init__(self, file_path, lang1, lang2, reverse=False):
        """
        Dataset preparation for neural machine translation.

        Args:
            file_path (str): Path to the dataset file.
            lang1 (str): Source language name.
            lang2 (str): Target language name.
            reverse (bool): Whether to reverse the source and target languages.
        """
        logger.info(f"Loading data from {file_path}")
        with open(file_path, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header row
            lines = list(reader)

        # Normalize and optionally reverse pairs
        self.pairs = [[utils.normalize_String(s) for s in line] for line in lines]
        if reverse:
            logger.info("Reversing targets")
            self.pairs = [list(reversed(p)) for p in self.pairs]

        # Initialize language vocabularies
        self.input_lang = utils.WordVocabulary(lang2 if reverse else lang1)
        self.output_lang = utils.WordVocabulary(lang1 if reverse else lang2)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError(f"Index {idx} out of range.")
        return self.pairs[idx]

    def get_languages(self):
        """Retrieve input and output language vocabularies."""
        return self.input_lang, self.output_lang


class NMTDataModule(pl.LightningDataModule):
    def __init__(self, file_path, lang1, lang2, split_ratio=0.8, batch_size=32, num_workers=2, max_len=12, min_len=2,seed=42, reverse=False):
        """
        Data module for NMT training and validation.

        Args:
            file_path (str): Path to the dataset file.
            lang1 (str): Source language name.
            lang2 (str): Target language name.
            split_ratio (float): Train-test split ratio.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of DataLoader workers.
            max_len (int): Maximum sentence length.
            min_len (int): Minimum sentence length.
            reverse (bool): Whether to reverse source and target languages.
        """
        super().__init__()
        self.file_path = file_path
        self.lang1 = lang1
        self.lang2 = lang2
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len, self.min_len = max_len, min_len
        self.seed = seed
        self.reverse = reverse
        self.input_lang = None
        self.output_lang = None
        self.pairs = None

    def _filter_and_prepare_data(self):
        """Load, filter, and prepare dataset for NMT."""
        prep_data = PrepData(self.file_path, self.lang1, self.lang2, self.reverse)
        self.input_lang, self.output_lang = prep_data.get_languages()
        pairs = utils.filterPairs(prep_data.pairs, self.max_len, self.min_len)
        
        logger.info(f"Read {len(prep_data)} sentence pairs")
        logger.info(f"Trimmed to {len(pairs)} sentence pairs | Max_len: {self.max_len}, Min_len: {self.min_len}")

        for pair in pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])

        logger.info(f"Input vocab: {self.input_lang.name} => {self.input_lang.n_words} words")
        logger.info(f"Target Vocab: {self.output_lang.name} => {self.output_lang.n_words} words")

        return pairs

    def setup(self, stage=None):
        """Setup datasets and create tokenized tensors."""
        self.pairs = self._filter_and_prepare_data()
        
        EOS_token = utils.EOS_Token
        num_pairs = len(self.pairs)

        input_ids = np.zeros((num_pairs, self.max_len), dtype=np.int32)
        target_ids = np.zeros((num_pairs, self.max_len), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(self.pairs):
            inp_ids = utils.indexesFromSentence(self.input_lang, inp)
            tgt_ids = utils.indexesFromSentence(self.output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        dataset = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))

        train_size = int(self.split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.train_sampler = RandomSampler(self.train_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        # NOTE: Used same val_dataloader for final_test without teacher forcing
        return self.val_dataloader()