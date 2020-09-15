#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/15 上午9:45
# @ Software   : PyCharm
#-------------------------------------------------------

import unicodedata
import re   # https://regex101.com/
import io
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

from libs.configs import cfgs



# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # w = '"he is a boy."'
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    # word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    """
    tokenize
    :param lang:
    :return:
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    sequence = lang_tokenizer.texts_to_sequences(lang)
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=sequence,
                                                           maxlen=None,
                                                           padding='post')
    return lang_tokenizer, sequence

def load_dataset(path, num_examples=None):
    input_lang, target_lang = create_dataset(path, num_examples=num_examples)

    input_tokenizer, input_sequence = tokenize(input_lang)
    target_tokenizer, target_sequence = tokenize(target_lang)

    save_to_pickle(cfgs.INPUT_WORD_INDEX, input_tokenizer.word_index)
    save_to_pickle(cfgs.TARGET_WORD_INDEX, target_tokenizer.word_index)

    seq_max_length = {
        'input_max_length': input_sequence.shape[-1],
        'target_max_length': target_sequence.shape[-1]
    }
    save_to_pickle(cfgs.SEQ_MAX_LENGTH, seq_max_length)


    return input_sequence, target_sequence


def split_dataset(input, target, split_ratio):
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input,
                                                                                                    target,
                                                                                                    test_size=split_ratio)
    return input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val


def dataset_batch(input, target, batch_size, epoch=None, shuffle=None):

    dataset = tf.data.Dataset.from_tensor_slices((input, target))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input))

    dataset = dataset.repeat(epoch).batch(batch_size, drop_remainder=True)

    return dataset


def save_to_pickle(filename, vocab):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
        return vocab


def index_to_word(tensor, index_word):
    text = ' '.join([index_word[t] for t in tensor if t != 0])
    return text


def word_to_index(text, word_index):
    text = preprocess_sentence(text)
    tensor = [word_index[t] for t in text]

    return tensor


if __name__ == "__main__":

    # Download the file
    # path_to_zip = tf.keras.utils.get_file(
    #     'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    #     extract=True)
    #
    # path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    # def load_parse_data(path):
    #     # -----------------------download dataset------------------------------------
    #     file_path = tf.keras.utils.get_file(path,
    #                                         origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    #
    #     # ----------------------read and decode data------------------------------------
    #     with open(file_path, 'rb') as f:
    #         text = f.read().decode(encoding='utf-8')
    #
    #     vocab = sorted(set(text))
    #     print('{} unique characters'.format(len(vocab)))
    #
    #     # get and save char index
    #     char_index = {char: index for index, char in enumerate(vocab)}
    #     index_char = np.array(vocab)
    #
    #     with open(cfgs.CHAR_INDEX, 'w') as f:
    #         f.write(json.dumps(char_index))
    #
    #     return text, vocab, char_index, index_char


    # en_sentence = u"May I borrow this book?"
    # sp_sentence = u"¿Puedo tomar prestado este libro?"
    # print(preprocess_sentence(en_sentence))
    # print(preprocess_sentence(sp_sentence).encode('utf-8'))



    # Try experimenting with the size of that dataset
    num_examples = 30000
    input_tensor, target_tensor = load_dataset(cfgs.DATASET_PATH, num_examples)

    # Calculate max_length of the target tensors
    # max_length_input, max_length_target,  = input_tensor.shape[1], target_tensor.shape[1]
    #
    # print('Input max length {}'.format(max_length_input))
    # print('Target max length {}'.format(max_length_target))
    # # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = split_dataset(input_tensor,
                                                                                                target_tensor,
                                                                                                split_ratio=cfgs.SPLIT_RATIO)
    # Show length
    # print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    # get word_index and index word
    input_word_index = read_from_pickle(cfgs.INPUT_WORD_INDEX)
    target_word_index = read_from_pickle(cfgs.TARGET_WORD_INDEX)

    input_index_word = {index:word for word, index in input_word_index.items()}
    target_index_word = {index: word for word, index in target_word_index.items()}

    # print("Input Language; index to word mapping")
    # for index, word in zip(input_tensor_train[0], index_to_word(input_tensor_train[0], input_index_word).split()):
    #     print("%d ----> %s" % (index, word))
    #
    # print("Target Language; index to word mapping")
    # for index, word in zip(target_tensor_train[0], index_to_word(target_tensor_train[0], target_index_word).split()):
    #     print("%d ----> %s" % (index, word))

    train_dataset = dataset_batch(input=input_tensor_train, target=target_tensor_train, batch_size=cfgs.BATCH_SIZE,
                                  shuffle=True)
    example_input_batch, example_output_batch = next(iter(train_dataset))
    print(example_input_batch.shape, example_output_batch.shape)








