#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/10 下午4:44
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import tensorflow as tf
import time

from data.dataset_pipeline import load_dataset, split_dataset, dataset_batch, read_word_index
from libs.configs import cfgs
from libs.nets.model import Encoder, Decoder, BahdanauAttention


if __name__ == "__main__":
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
    input_word_index = read_word_index(cfgs.INPUT_WORD_INDEX)
    target_word_index = read_word_index(cfgs.TARGET_WORD_INDEX)

    input_index_word = {index: word for word, index in input_word_index.items()}
    target_index_word = {index: word for word, index in target_word_index.items()}

    vocab_size_input = len(input_index_word)
    vocab_size_target = len(target_index_word)

    train_dataset = dataset_batch(input=input_tensor_train, target=target_tensor_train, batch_size=cfgs.BATCH_SIZE,
                                  shuffle=True)
    # example_input_batch, example_output_batch = next(train_dataset)
    #
    encoder = Encoder(batch_size=cfgs.BATCH_SIZE, vocab_size=vocab_size_input, embedding_dim=cfgs.EMBEDDING_DIM,
                      encode_units=cfgs.NUM_UNITS)

    # # sample input
    # sample_hidden = encoder.initialize_hidden_state()
    #
    # sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    # print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    #
    # attention_layer = BahdanauAttention(10)
    # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    #
    # print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    #
    decoder = Decoder(vocab_size_target, cfgs.EMBEDDING_DIM, cfgs.NUM_UNITS, cfgs.BATCH_SIZE)
    #
    # sample_decoder_output, _, _ = decoder(tf.random.uniform((cfgs.BATCH_SIZE, 1)),
    #                                       sample_hidden, sample_output)
    #
    # print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))



    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    checkpoint_prefix = os.path.join(cfgs.TRAINED_CKPT, "ckpt_{epoch}")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)


    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([target_word_index['<start>']] * cfgs.BATCH_SIZE, axis=1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    steps_per_epoch = len(input_tensor_train) // cfgs.BATCH_SIZE

    for epoch in range(cfgs.NUM_EPOCH):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
