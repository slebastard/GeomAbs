## Testing libraries imports
import pytest
import pdb

## Standard library imports 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## Home-cooked methods imports
import os, sys
lib_path = os.path.abspath('./methods')
sys.path.insert(0, lib_path)


##########################################

def make_dataset():
	from keras.datasets import mnist
	from data_tools import ImageDataset

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	filt_labels = range(8)
	return ImageDataset(x_train, y_train, x_test, y_test, filt_labels=filt_labels, spl=0.06)

def test_make_dataset():
	ds = make_dataset()
	assert True

def test_mnist_build_a_dichotomy():
	mnist = make_dataset()
	mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
	mnist.build_dichLabels(mnist_parity, 'parity')

	mnist_smallness = [range(0,4), range(4,8)]
	mnist.build_dichLabels(mnist_smallness, 'smaller_than_4')
	assert True

def test_mnist_hstack_dichotomies():
	mnist = make_dataset()
	# Currently dichotomies will only be binary
	mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
	mnist_smallness = [range(0,4), range(4,8)]
	mnist.build_dichLabels(mnist_smallness, 'smaller_than_4')
	mnist.build_dichLabels(mnist_parity, 'parity')
	mnist.hstack_dichs('parity', 'smaller_than_4')
	assert True

def test_mnist_product_dichotomies():
	mnist = make_dataset()
	# Currently dichotomies will only be binary
	mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
	mnist_smallness = [range(0,4), range(4,8)]
	mnist_prod = [set(s1).intersection(set(s2)) for s2 in mnist_smallness for s1 in mnist_parity]
	mnist.build_dichLabels(mnist_smallness, 'smaller_than_4')
	mnist.build_dichLabels(mnist_parity, 'parity')
	mnist.hstack_dichs('parity', 'smaller_than_4')
	assert True

def test_mnist_compstack_dichotomies():
	mnist = make_dataset()
	# Currently dichotomies will only be binary
	mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
	mnist_smallness = [range(0,4), range(4,8)]
	mnist.build_dichLabels(mnist_smallness, 'smaller_than_4')
	mnist.build_dichLabels(mnist_parity, 'parity')
	mnist_prod = [set(s1).intersection(set(s2)) for s2 in mnist_smallness for s1 in mnist_parity]
	mnist.build_dichLabels(mnist_prod, 'parity_prod_smaller_than_4')
	assert True