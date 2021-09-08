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
from src import ConfigClass, Mnist_SVDD

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

mnist = Mnist_SVDD(config)
#mnist.fit()
mnist.test()
