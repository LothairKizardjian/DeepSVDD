import argparse
import os
import sys
import ast
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.mnist import load_data

from src import mnist_cnn, ConfigClass

###########################
#####     PARSER      #####
###########################

parser = argparse.ArgumentParser(
    description="Implementatino of Deep SVDD paper."
)

parser.add_argument(
    "-c", "--configfile", type=str, help="path of the config file", required=True
)

args = parser.parse_args()
config_file_path = args.configfile
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file_path)
config = ConfigClass(config)

###########################
#####      MAIN       #####
###########################

(x_train, y_train), (x_test, y_test) = load_data()
x_val = x_train[:10000]
y_val = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

model = mnist_cnn(
    input_shape = config.input_shape,
    output_channels = config.output_channels,
    activation = config.activation,
    dropout = config.dropout,
    pool_size = config.pool_size,
    filt_size = config.filt_size,
    kernel_size = config.kernel_size)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=config.batch_size,
          validation_data=(x_val, y_val), epochs=config.epochs, verbose=1)
