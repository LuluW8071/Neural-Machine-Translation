import pytorch_lightning as pl
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
            reader = csv.reader(csvfile, delimiter='\t')
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
    def __init__(self, train_path, valid_path, lang1, lang2, batch_size=32, num_workers=2, max_len=12, min_len=2, reverse=False):
        """
        Data module for NMT training, validation, and testing.

        Args:
            train_path (str): Path to the training dataset file.
            valid_path (str): Path to the validation dataset file.
            lang1 (str): Source language name.
            lang2 (str): Target language name.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of DataLoader workers.
            max_len (int): Maximum sentence length.
            min_len (int): Minimum sentence length.
            reverse (bool): Whether to reverse source and target languages.
        """
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.lang1 = lang1
        self.lang2 = lang2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len, self.min_len = max_len, min_len
        self.reverse = reverse
        self.input_lang = None
        self.output_lang = None
        self.train_pairs = None
        self.valid_pairs = None

    def _filter_and_prepare_data(self, file_path):
        """Load, filter, and prepare dataset for NMT."""
        prep_data = PrepData(file_path, self.lang1, self.lang2, self.reverse)
        self.input_lang, self.output_lang = prep_data.get_languages()
        pairs = utils.filterPairs(prep_data.pairs, self.max_len, self.min_len)
        
        logger.info(f"Read {len(prep_data)} sentence pairs from {file_path}")
        logger.info(f"Trimmed to {len(pairs)} sentence pairs | Max_len: {self.max_len}, Min_len: {self.min_len}")

        for pair in pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])

        logger.info(f"Input vocab: {self.input_lang.name} => {self.input_lang.n_words} words")
        logger.info(f"Target Vocab: {self.output_lang.name} => {self.output_lang.n_words} words")

        return pairs
    
    def _collect_vocab_from_pairs(self, combined_pairs, input_lang, output_lang):
        """
        Collect vocabulary from combined sentence pairs and update input/output vocabularies.
        
        Args:
            combined_pairs (list): List of sentence pairs from train, valid, and test sets.
            input_lang (WordVocabulary): Vocabulary for the input language.
            output_lang (WordVocabulary): Vocabulary for the output language.
        """
        for pair in combined_pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

    def _prepare_data_pairs(self, pairs, input_lang, output_lang, max_len, EOS_token):
        """
        Prepares input and target indices for a dataset.
        
        Args:
            pairs (list): List of sentence pairs.
            input_lang (WordVocabulary): Vocabulary for the input language.
            output_lang (WordVocabulary): Vocabulary for the output language.
            max_len (int): Maximum sentence length.
            EOS_token (int): End-of-sentence token.
            
        Returns:
            tuple: Tuple of input_ids and target_ids arrays.
        """
        num_pairs = len(pairs)
        input_ids = np.zeros((num_pairs, max_len), dtype=np.int32)
        target_ids = np.zeros((num_pairs, max_len), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = utils.indexesFromSentence(input_lang, inp)
            tgt_ids = utils.indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        return input_ids, target_ids

    def setup(self, stage=None):
        """Setup datasets for train, validation, and test."""
        # Filter and prepare the data for each of the datasets
        self.train_pairs = self._filter_and_prepare_data(self.train_path)
        self.valid_pairs = self._filter_and_prepare_data(self.valid_path)

        EOS_token = utils.EOS_Token

        # Combine all the pairs from train, valid, and test datasets
        combined_pairs = self.train_pairs + self.valid_pairs

        # Collect vocab from the combined dataset
        self._collect_vocab_from_pairs(combined_pairs, self.input_lang, self.output_lang)

        # Save vocabularies
        self.input_lang.save_to_file("input_vocab.json", input=True)
        self.output_lang.save_to_file("output_vocab.json", input=False)

        logger.info(f"Input vocab size: {self.input_lang.n_words}")
        logger.info(f"Output vocab size: {self.output_lang.n_words}")

        # Prepare data pairs for each split using the refactored function
        input_ids, target_ids = self._prepare_data_pairs(combined_pairs, self.input_lang, self.output_lang, self.max_len, EOS_token)

        # Now, split the input_ids and target_ids back into train, valid, and test sets
        num_train = len(self.train_pairs)
        num_valid = len(self.valid_pairs)

        # Split the data back into respective datasets
        self.train_input_ids = input_ids[:num_train]
        self.train_target_ids = target_ids[:num_train]

        self.valid_input_ids = input_ids[num_train:num_train + num_valid]
        self.valid_target_ids = target_ids[num_train:num_train + num_valid]

        # Create TensorDatasets for each split
        self.train_dataset = TensorDataset(torch.LongTensor(self.train_input_ids), torch.LongTensor(self.train_target_ids))
        self.valid_dataset = TensorDataset(torch.LongTensor(self.valid_input_ids), torch.LongTensor(self.valid_target_ids))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
