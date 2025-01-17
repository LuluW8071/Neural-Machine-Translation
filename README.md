# Neural Machine Translation

<div align="center">

![Status](https://img.shields.io/badge/status-completed-green.svg) ![License](https://img.shields.io/github/license/LuluW8071/Neural-Machine-Translation) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Neural-Machine-Translation) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Neural-Machine-Translation) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Neural-Machine-Translation)

</div>

This repository provides an implementation of the paper *__Neural Machine Translation by Jointly Learning to Align and Translate__*. It includes training scripts with support for distributed GPU training using **Lightning AI**.

## Usage

#### 1. Install Dependencies  

```bash
sudo apt-get install -y fonts-noto-core fonts-noto-unhinted fonts-noto-hinted libcairo2 libcairo2-dev libpango1.0-dev libglib2.0-dev
pip install -r requirements.txt
```

#### 2. Prepare the Dataset

> [!NOTE]  
> The dataset script reads the dataset file in __TSV__ format.

Download the dataset from [__here__](https://drive.google.com/drive/folders/1iqhugLp0wwZau5Q2p5ju-z__C0t6XqCV?usp=sharing), and obtain the Nepali font from [__here__](https://nepalifonts.net/fonts/mangal-regular).

#### 3. Train the Model  
Before starting, add your **Comet ML API key** and **project name** to the `.env` file.

| **Argument**          | **Description** | **Default Value** |
|----------------------|-----------------|------------------|
| `-d`, `--device` | Device for training | `cuda` |
| `-g`, `--gpus` | Number of GPUs per node | `1` |
| `-w`, `--num_workers` | Number of data loading workers | `8` |
| `-db`, `--dist_backend` | Distributed backend for multi-GPU training | `ddp` |
| `--train_path` | Path to the training dataset (TSV format) | None (required) |
| `--valid_path` | Path to the validation dataset (CSV format) | None (required) |
| `--font_path` | Path to the font file (.ttf), useful for Devanagari scripts | None (required) |
| `--input_lang` | Source language | `np` |
| `--output_lang` | Target language | `en` |
| `--reverse` | Reverse source and target languages | `False` |
| `--max_len` | Maximum sequence length | `12` |
| `--min_len` | Minimum sequence length | `2` |
| `-mt`, `--model_type` | Model type: `lstm` or `gru` | `lstm` |
| `-hs`, `--hidden_size` | Model hidden size | `128` |
| `-nl`, `--num_layers` | Number of model layers | `2` |
| `-bd`, `--bidirection` | Use bidirectional model | `False` |
| `-at`, `--attention` | Use attention mechanism | `False` |
| `--epochs` | Number of training epochs | `100` |
| `--batch_size` | Training batch size | `64` |
| `-lr`, `--learning_rate` | Initial learning rate | `4e-4` |
| `-lrf`, `--lr_factor` | Factor for learning rate decay | `0.5` |
| `-lrp`, `--lr_patience` | Learning rate patience for decay | `1` |
| `-mlt`, `--min_lr_threshold` | Minimum learning rate threshold | `1e-2` |
| `-mlr`, `--min_lr` | Minimum learning rate | `1e-4` |
| `--precision` | Precision mode | `32-true` |
| `--checkpoint_path` | Path to a checkpoint file for resuming training | None |
| `-gc`, `--grad_clip` | Gradient norm clipping value | `1.0` |
| `-ag`, `--acc_grad` | Number of batches for gradient accumulation | `2` |

```bash
python3 input/train.py \
  -d cuda -w 4 -g 2 -db ddp \
  --input_lang ne --output_lang en \
  --train_path train.tsv --valid_path valid.tsv --font_path custom_font.ttf \
  --batch_size 64 -ag 2 --epochs 50 --max_len 16 --min_len 4 \
  -lr 4e-3 -mlr 1e-4 \
  -hs 256 --nl 2 --mt lstm --attention --bidirection
```

> [!NOTE]  
> Use the `--reverse` flag if you want to train the model with reversed source and target languages.

## Experiment Results

The dataset was filtered with a `max_len` of __16__ and a `min_len` of __4__. A total of ~500k sentence pairs were used for training and ~90k for validation. The model was trained on 2×T4 GPUs with a batch size of 128 (Effective Batch Size = 2×128). __Sacre BLEU score__ with __4-gram__ was used as the primary evaluation metric.

> [!IMPORTANT]  
> Teacher forcing helps train variable-length sequences by preventing error propagation and ensuring stable learning, as cross-entropy loss struggles with varying lengths. So, targets was paased during training and validation. For final validation (test), inference is done without teacher forcing to test real-world performance.  

### Nepali to English  

| **Encoder** | **Decoder** | **Attention** | **RNN Layers** | **Hidden Size** | **Avg. BLEU Score** |  
|-------------|------------|---------------|----------------|-----------------|----------------|  
| GRU         | GRU        | False         | 2              | 128             | 10.758         |  
| LSTM        | LSTM       | True          | 2              | 128             | 22.598         |  
| BiLSTM      | LSTM       | True          | 2              | 128             | 25.155         |  

#### Translations  

> Examples of translations from Nepali to English using __BiLSTM-LSTM with Attention__.   

| **Source** | **Ground Truth** | **Translated** | **BLEU Score** |  
|------------|----------------|----------------|----------------|  
| काठमाडौं पोष्ट अक्टोबर ५ २०१९ बाट केही माथिल्ला कथाहरू यहाँ छन् । | here are some of the top stories from the kathmandu post october 5 2019 . | here are some top stories from kathmandu post october 5 2019 2019 . | 47.83 |  
| जुलुसमा नेपाली कांग्रेस गणेशमान र गिरिजा मुर्दावादका नारा लगाएका थिए । | the procession chanted slogans of nepali congress ganeshman and girija murdabad . | in the procession the slogans of nepali congress ganeshman and girija prasad . | 57.27 |  
| कार्यक्रम सफल पार्न जनसम्पर्क समितिले सबैमा आव्हान गरेको छ | the public relations committee has called for everyone to succeed the program | the public committee has invited the public relations to succeed the program | 42.23 |  
| नेपाल विद्युत् प्राधिकरणले विभिन्न परियोजनाहरूकालागि प्रस्ताव गरेको विद्युत् खरिद सम्झ्ौतालाई परिमार्जन गर्नुपर्छ । | the nepal electricity authority should modify the power purchase agreement proposed for various projects . | nepal electricity authority should propose that the electricity purchased by various syndicates . | 25.26 |  

#### Attention Plots  

| **Translation 1** | **Translation 2** |  
|------------------|------------------|  
| ![BiLSTM-LSTM-Attn1](assets/attn_plot_1.png) | ![BiLSTM-LSTM-Attn2](assets/attn_plot_2.png) |  
| **Translation 3** | **Translation 4** |  
| ![BiLSTM-LSTM-Attn3](assets/attn_plot_3.png) | ![BiLSTM-LSTM-Attn4](assets/attn_plot_4.png) | 

### English to Nepali

| **Encoder** | **Decoder** | **Attention** | **RNN Layers** | **Hidden Size** | **BLEU Score** |
|-------------|-------------|---------------|----------------|-----------------|----------------|
| Bi-LSTM     | LSTM        | True          | 2              | 128             | 15.070         |


---

## Citation

```bibtex
@inproceedings{bahdanau2015neural,
  title={Neural Machine Translation by Jointly Learning to Align and Translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  booktitle={3rd International Conference on Learning Representations, {ICLR} 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings},
  year={2015},
  archivePrefix={arXiv},
  eprint={1409.0473},
  primaryClass={cs.CL},
  url={https://doi.org/10.48550/arXiv.1409.0473}
}
```
