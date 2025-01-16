import re
import random
import json
import torch
import numpy as np

SOS_Token = 0
EOS_Token = 1

class WordVocabulary:
    """ Word class to store vocabulary and corpus """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2  # Count SOS and EOS Tokens

    def addSentence(self, sentence):
        """ Split sentence into words and add to vocabulary """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """ Function: Add word to vocabulary if not added previously"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def save_to_file(self, file_path, input=True):
        """Save vocabulary to a file. Save word2index if input=True, else save index2word."""
        if input:
            data = {
                "name": self.name,
                "word2index": self.word2index,
                "n_words": self.n_words
            }
        else:
            data = {
                "name": self.name,
                "index2word": self.index2word,
                "n_words": self.n_words
            }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path, input=True):
        """Load vocabulary from a file. Load word2index if input=True, else load index2word."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = cls(data["name"])
        if input:
            vocab.word2index = data["word2index"]
        else:
            vocab.index2word = data["index2word"]

        vocab.n_words = data["n_words"]
        print(f"Vocabulary loaded from {file_path}")
        return vocab

# NOTE: Might need change in the pre processing of strings as per language used
def normalize_String(s):
    """ Lowercase, trim and remove non-letter characters """
    s = s.lower().strip()

    # Replace multiple punctuation marks (e.g., "..", "...") with a single punctuation mark
    s = re.sub(r"([.!?।])\1+", r"\1", s)

    # Remove all commas
    s = s.replace(",", "")

    # Add a space before punctuation marks if not already present (except for apostrophes)
    s = re.sub(r"(?<!\s)([.!?¿¡।])", r" \1", s)

    # Remove patterns like (1), (२), or (ग)
    s = re.sub(r"\(\d+\)|\([\u0966-\u096F]+\)|\([a-z\u0900-\u097F]\)", r"", s)

    # Retain Devanagari, English & Latin characters, numbers, punctuation, and apostrophes; replace others with a space
    s = re.sub(r"[^\u0900-\u097Fa-zA-Z0-9.!?'\u2019]+", r" ", s)

    # Remove extra spaces
    s = re.sub(r"\s+", r" ", s).strip()

    return s




def filterPairs(pairs, max_len=32, min_len=4):
    """ Filter pairs of sentences with length greater than max_len and less than min_len """
    MAX_LENGTH, MIN_LENGTH = max_len, min_len
    return [
        pair for pair in pairs
        if (len(pair[0].split(' ')) >= MIN_LENGTH and len(pair[1].split(' ')) >= MIN_LENGTH) and
        (len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH)
    ]


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_Token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)


def sentenceFromIndexes(lang, indexes):
    """Convert a list of token indices back into a sentence."""
    words = []
    for index in indexes:
        if index in lang.index2word:
            word = lang.index2word[index]
            # Stop at the EOS token
            if word == "<EOS>":
                break
            words.append(word)
    return ' '.join(words)


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    # NOTE: Commented reproducibility seeding due to teacher forcing probablity in the model
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Make the results deterministic
    # torch.backends.cudnn.benchmark = False     # Disable auto-tuner to ensure deterministic behavior