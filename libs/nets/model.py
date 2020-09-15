#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/15 下午3:25
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, batch_size, vocab_size, embedding_dim, encode_units):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encode_units = encode_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.encode_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, input, hidden_state):
        """
        :param input:  (batch_size, sequence_length)
        :param hidden_state: (batch_size, num_units)
        :return:
        """
        # x => (batch_size, sequence length, embedding_dim)
        x = self.embedding(input)

        # output => (batch size, sequence length, units)
        # state => (batch_size, num_units)
        output, state = self.gru(x, initial_state=hidden_state)

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.encode_units))


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values):
        # query hidden state shape == (batch_size, hidden_size)
        # query_with_time_axis shape == (batch_size, 1, hidden_size)
        # values shape == (batch_size, max_len, hidden_size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        # query_with_time_axis => (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, axis=1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # self.W1(query_with_time_axis) =>  (batch_size, 1, units)
        # self.W2(values) =>  (batch_size, max_len, units)
        # score => (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def __call__(self, x, hidden, enc_output):
        """

        :param x: (batch_size, 1)
        :param hidden: (batch_size, num_units)
        :param enc_output: (batch_size, max_length, num_units)
        :return:
        """
        # context_vector => (batch_size, num_units)
        # attention_weights => (batch_size, max_length, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # output => (batch_size, 1, hidden_size)
        # state => (batch_size, hidden_size)
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

