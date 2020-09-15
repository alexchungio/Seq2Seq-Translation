#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/15 上午9:50
# @ Software   : PyCharm
#-------------------------------------------------------

from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
VERSION = 'Seq2Seq_Translation_20200915'
NET_NAME = 'seq2seq_translation'


#------------------------------GPU config
# ------------get gpu and cpu list------------------
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus)
# print(cpus)

# ------------------set visible of current program-------------------
# method 1 Terminal input
# $ export CUDA_VISIBLE_DEVICES = 2, 3
# method 1
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# method 2
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# ----------------------set gpu memory allocation-------------------------
# method 1: set memory size dynamic growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# method 2: set allocate static memory size
# tf.config.experimental.set_virtual_device_configuration(
#     device=gpus[0],
#     logical_devices = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
# )


# ---------------------------------------- System_config----------------------------
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"
SHOW_TRAIN_INFO_INTE = 100
SMRY_ITER = 100
SAVE_WEIGHTS_ITER = 5

SUMMARY_PATH = ROOT_PATH + '/outputs/summary'
INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'
# INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

TRAINED_CKPT = os.path.join(ROOT_PATH, 'outputs/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'

INPUT_WORD_INDEX = ROOT_PATH + '/outputs/input_word_index.pickle'
TARGET_WORD_INDEX = ROOT_PATH + '/outputs/target_word_index.pickle'
SEQ_MAX_LENGTH = ROOT_PATH + '/outputs/seq_max_length.pickle'

#----------------------Data---------------------
DATASET_PATH = os.path.join(ROOT_PATH, 'data', 'spa-eng', 'spa.txt')


#------------------------network config--------------------------------
BATCH_SIZE = 64

SEQUENCE_LENGTH = 100 # the number in singe time dimension of a single sequence of input data
VOCAB_SIZE = 26
EMBEDDING_DIM = 256

# NUM_UNITS = [128, 64, 32]
NUM_UNITS = 1024


#-------------------------train config-------------------------------
EMBEDDING_TRANSFER = False
LEARNING_RATE = 0.001
NUM_EPOCH = 10
KEEP_PROB = 1.0

# data
SPLIT_RATIO = 0.2