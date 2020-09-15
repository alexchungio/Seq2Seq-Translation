#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/15 下午4:52
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf


from libs.configs import cfgs
from data.dataset_pipeline import load_dataset, split_dataset, dataset_batch, read_from_pickle, preprocess_sentence
from libs.nets.model import Encoder, Decoder


input_word_index = read_from_pickle(cfgs.INPUT_WORD_INDEX)
target_word_index = read_from_pickle(cfgs.TARGET_WORD_INDEX)

input_index_word = {index: word for word, index in input_word_index.items()}
target_index_word = {index: word for word, index in target_word_index.items()}

vocab_size_input = len(input_index_word)
vocab_size_target = len(target_index_word)


seq_max_length = read_from_pickle(cfgs.SEQ_MAX_LENGTH)
max_length_input, max_length_target = seq_max_length['input_max_length'], seq_max_length['target_max_length']

def evaluate(sentence, ):
    attention_plot = np.zeros((max_length_target, max_length_input))

    sentence = preprocess_sentence(sentence)

    inputs = [input_word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_input,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, cfgs.NUM_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_word_index['<start>']], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_index_word[predicted_id] + ' '

        if target_index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))



if __name__ == "__main__":


    # example_input_batch, example_output_batch = next(train_dataset)
    #
    encoder = Encoder(batch_size=cfgs.BATCH_SIZE, vocab_size=vocab_size_input, embedding_dim=cfgs.EMBEDDING_DIM,
                      encode_units=cfgs.NUM_UNITS)

    #
    decoder = Decoder(vocab_size_target, cfgs.EMBEDDING_DIM, cfgs.NUM_UNITS, cfgs.BATCH_SIZE)
    #
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix = os.path.join(cfgs.TRAINED_CKPT, "ckpt_{epoch}")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(cfgs.TRAINED_CKPT))


    English_text = u'This is too difficult for me.'
    translate(English_text)
