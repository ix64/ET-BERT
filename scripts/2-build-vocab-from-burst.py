#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import os
import tempfile

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def build_BPE(dst_wordpiece_path: str, src_corpora_file: str):
    # generate source dictionary,0-65535
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {}
    tuple_sep = ()
    tuple_cls = ()
    # 'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:04x}'.format(i)
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary, unk_token="[UNK]", max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]", 1), cls=('[CLS]', 2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([src_corpora_file, src_corpora_file], trainer=trainer)

    # And Save it
    tokenizer.save(dst_wordpiece_path, pretty=True)
    return 0


def build_vocab(dst_vocab_path: str, src_wordpiece_path: str, ):
    with open(src_wordpiece_path, 'r') as json_file:
        vocab_json = json.load(json_file)
    vocab_txt = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item)  # append key of vocab_json
    with open(dst_vocab_path, 'w') as f:
        for word in vocab_txt:
            f.write(word + "\n")


def task_build_vocab(dst_vocab_path: str, src_corpora_path: str):
    wordpiece_path = tempfile.mktemp()
    build_BPE(wordpiece_path, src_corpora_path)
    build_vocab(dst_vocab_path, wordpiece_path)
    os.remove(wordpiece_path)


if __name__ == '__main__':
    word_filepath = r"D:\dev\clone\ET-BERT\download\encrypted_traffic_burst.txt"
    vocab_filepath = r"D:\dev\clone\ET-BERT\models\vocab.txt"
    task_build_vocab(vocab_filepath, word_filepath)
