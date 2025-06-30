import re
import random
import json
import torch
import numpy as np
import warnings
import sentencepiece as spm
import os

warnings.filterwarnings("ignore", category=UserWarning)

SOS_Token = 0
EOS_Token = 1

class WordVocabulary:
    """BPE vocabulary using SentencePiece."""
    def __init__(self, name, model_prefix=None, input_text_path=None, vocab_size=16000):
        self.name = name
        self.sp = spm.SentencePieceProcessor()
        self.model_prefix = model_prefix or name
        model_file = f"{self.model_prefix}.model"
        vocab_file = f"{self.model_prefix}.vocab"

        if os.path.exists(model_file) and os.path.exists(vocab_file):
            # Load existing model
            self.sp.load(model_file)
            print(f"[INFO] Loaded existing SentencePiece model: {model_file}")
        elif input_text_path:
            print(f"[INFO] Training new SentencePiece model: {model_file}")

            # Choose character coverage based on model_prefix
            if "en" in self.model_prefix.lower():
                character_coverage = 0.9995  # English (ASCII-dominant)
            else:
                character_coverage = 1.0     # Nepali or other non-Latin languages

            spm.SentencePieceTrainer.train(
                input=input_text_path,
                model_prefix=self.model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                bos_id=SOS_Token,          # <s>
                eos_id=EOS_Token,          # </s>
                pad_id=2,          # <pad>
                unk_id=3,          # <unk>
                pad_piece="<pad>",
                bos_piece="<s>",
                eos_piece="</s>",
                unk_piece="<unk>",
                character_coverage=character_coverage,
                train_extremely_large_corpus=True,
                hard_vocab_limit=False
            )
            self.sp.load(model_file)
        else:
            raise ValueError("Must provide either both model files or a corpus to train.")

    def addSentence(self, sentence):
        pass  # Not needed in BPE

    def addWord(self, word):
        pass  # Not needed in BPE

    @property
    def word2index(self):
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}

    @property
    def index2word(self):
        return {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}

    @property
    def n_words(self):
        return self.sp.get_piece_size()

    def save_to_file(self, file_path, input=True):
        data = {
            "name": self.name,
            "model_prefix": self.model_prefix
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path, input=True):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data["name"], model_prefix=data["model_prefix"])

# NOTE: Might need change in the pre processing of strings as per language used
def normalize_String(s):
    """ Lowercase, trim and remove non-letter characters """
    s = s.lower().strip()

    # Replace multiple punctuation marks (e.g., "..", "...") with a single punctuation mark
    s = re.sub(r"([.!?।])\1+", r"\1", s)

    # Remove all commas
    # s = s.replace(",", "")

    # Add a space before punctuation marks if not already present (except for apostrophes)
    # s = re.sub(r"(?<!\s)([.!?¿¡।])", r" \1", s)

    # Remove patterns like (1), (२), or (ग)
    s = re.sub(r"\(\d+\)|\([\u0966-\u096F]+\)|\([a-z\u0900-\u097F]\)", r"", s)

    # Retain Devanagari, English & Latin characters, numbers, punctuation, and apostrophes; replace others with a space
    # s = re.sub(r"[^\u0900-\u097Fa-zA-Z0-9.!?'\u2019]+", r" ", s)

    # Remove extra spaces
    s = re.sub(r"\s+", r" ", s).strip()

    return s



def filterPairs(pairs, max_len=99, min_len=4, tokenizer=None):
    return [
        pair for pair in pairs
        if (len(tokenizer.sp.encode(pair[0])) >= min_len and len(tokenizer.sp.encode(pair[0])) <= max_len) and
           (len(tokenizer.sp.encode(pair[1])) >= min_len and len(tokenizer.sp.encode(pair[1])) <= max_len)
    ]


def indexesFromSentence(lang, sentence):
    return lang.sp.encode(sentence, out_type=int)

def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_Token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)

def sentenceFromIndexes(lang, indexes):
    tokens = [lang.sp.id_to_piece(i) for i in indexes if i != EOS_Token]
    return lang.sp.decode_pieces(tokens)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    # NOTE: Commented reproducibility seeding due to teacher forcing probablity in the model
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Make the results deterministic
    # torch.backends.cudnn.benchmark = False     # Disable auto-tuner to ensure deterministic behavior