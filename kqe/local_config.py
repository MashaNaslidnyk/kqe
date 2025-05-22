import platform
from os import path

ROOTDIR = None

DATADIR = path.join(ROOTDIR, "data")
FIGSDIR = path.join(ROOTDIR, "figs")
LOGSDIR = path.join(ROOTDIR, "logs")
MNISTDIR = path.join(ROOTDIR, "data", "MNIST_features")
MNIST77DIR = path.join(ROOTDIR, "data", "MNIST_7x7")
