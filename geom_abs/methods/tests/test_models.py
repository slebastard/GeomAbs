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


# Loading small sample of MNIST dataset
def build_mnist_ds(filt_lbls=range(8), spl=0.04):
	from keras.datasets import mnist
	from data_tools import ImageDataset

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	filt_labels = range(8)
	mnist = ImageDataset(x_train, y_train, x_test, y_test, filt_labels=filt_lbls, spl=spl)
	mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
	mnist_smallness = [range(0,4), range(4,8)]
	mnist_prod = [set(s1).intersection(set(s2)) for s2 in mnist_smallness for s1 in mnist_parity]
	mnist.build_dichLabels(mnist_smallness, 'smaller_than_4')
	mnist.build_dichLabels(mnist_parity, 'parity')
	mnist.hstack_dichs('parity', 'smaller_than_4')
	mnist.compstack_dichs('parity', 'smaller_than_4')
	mnist.build_dichLabels(mnist_prod, 'parity_prod_smaller_than_4')

	return mnist

def test_build_mnist_ds():
	mnist = build_mnist_ds()
	assert True

mnist_8 = build_mnist_ds(filt_lbls=range(8))
mnist_10 = build_mnist_ds(filt_lbls=None)

# Test instances methods #################################################

def build_model_for_vanilla_task():
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from models import Model

	(w_in, w_1, w_2, w_out) = (mnist_10.tot_dim, 100, 100, 10)
	max_epochs = 400
	model_arch = Sequential([
		Dense(w_1, input_shape=(w_in,)),
		Activation('tanh'),
		Dense(w_2),
		Activation('tanh'),
		Dense(w_out),
		Activation('softmax')
	])
	return Model(model_arch)

def build_model_for_hstack_dichs():
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from models import Model

	(w_in, w_1, w_2, w_out) = (mnist_8.tot_dim, 100, 100, 4)
	max_epochs = 400
	max_epochs = 400
	model_arch = Sequential([
		Dense(w_1, input_shape=(w_in,)),
		Activation('tanh'),
		Dense(w_2),
		Activation('tanh'),
		Dense(w_out),
		Activation('softmax')
	])
	return Model(model_arch)

def build_model_for_prod_dichs():
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from models import Model

	(w_in, w_1, w_2, w_out) = (mnist_8.tot_dim, 100, 100, 4)
	max_epochs = 400
	max_epochs = 400
	model_arch = Sequential([
		Dense(w_1, input_shape=(w_in,)),
		Activation('tanh'),
		Dense(w_2),
		Activation('tanh'),
		Dense(w_out),
		Activation('softmax')
	])
	return Model(model_arch)

def build_model_for_compstack_dichs():
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from models import Model

	(w_in, w_1, w_2, w_out) = (mnist_8.tot_dim, 100, 100, 2)
	max_epochs = 400
	max_epochs = 400
	model_arch = Sequential([
		Dense(w_1, input_shape=(w_in,)),
		Activation('tanh'),
		Dense(w_2),
		Activation('tanh'),
		Dense(w_out),
		Activation('softmax')
	])
	return Model(model_arch)

###################################################################

@pytest.mark.skip(reason="too time-expensive")
def test_model_on_vanilla_task():
	test_model = build_model_for_vanilla_task()
	test_model.fit(mnist_10, epochs=10, batch_size=32)
	test_model.evaluate(mnist_10, batch_size=128)
	assert True

def test_model_on_hstack_dichs():
	test_model = build_model_for_hstack_dichs()
	test_model.fit(mnist_8, dich_name='parity_hstack_smaller_than_4', epochs=10, batch_size=32)
	test_model.evaluate(mnist_8, dich_name='parity_hstack_smaller_than_4', batch_size=128)
	assert True

def test_model_on_prod_dichs():
	test_model = build_model_for_prod_dichs()
	test_model.fit(mnist_8, dich_name='parity_prod_smaller_than_4', epochs=10, batch_size=32)
	test_model.evaluate(mnist_8, dich_name='parity_prod_smaller_than_4', batch_size=128)
	assert True

@pytest.mark.skip(reason="too time-expensive")
def test_model_on_compstack_dichs():
	test_model = build_model_for_compstack_dichs()
	test_model.fit(mnist_8, dich_name='parity_compstack_smaller_than_4', epochs=10, batch_size=32)
	test_model.evaluate(mnist_8, dich_name='parity_compstack_smaller_than_4', batch_size=128)
	assert True

hstack_test_model = build_model_for_hstack_dichs()
hstack_test_model.fit(mnist_8, dich_name='parity_hstack_smaller_than_4', epochs=10, batch_size=32)

def test_model_sample_visualization():
	hstack_test_model.sample_eval(mnist_8, 1)
	hstack_test_model.sample_eval(mnist_8, 5)
	assert True

###################################################################

def test_get_repr():
	from sklearn.decomposition import PCA

	rprs_vanilla = hstack_test_model.get_repr(mnist_8, mnist_8.spl)

	pca2 = PCA(n_components=2)
	rprs_pca2 = hstack_test_model.get_repr(mnist_8, mnist_8.spl, pca2)

	pca3 = PCA(n_components=3)
	rprs_pca3 = hstack_test_model.get_repr(mnist_8, mnist_8.spl, pca3)

	fig1 = hstack_test_model.plot_reprs(mnist_8, mnist_8.spl, pca2)
	fig2 = hstack_test_model.plot_reprs(mnist_8, mnist_8.spl, pca3)

	assert True

def test_get_all_PS():
	rprs = hstack_test_model.get_repr(mnist_8, mnist_8.spl)

	PS1 = hstack_test_model.get_all_PS(mnist_8, rprs, lay_id=1)
	PS3 = hstack_test_model.get_all_PS(mnist_8, rprs, lay_id=3)
	PS5 = hstack_test_model.get_all_PS(mnist_8, rprs, lay_id=5)

	assert True

def test_get_dich_CCGP():
	CCGP_across_layers = hstack_test_model.get_dich_CCGP(mnist_8, (0, 1, 2, 3), n_labels_retained=2)
	CCGP_across_layers = hstack_test_model.get_dich_CCGP(mnist_8, (0, 2, 4, 6), n_labels_retained=1)
	assert True

def test_get_all_CCGP():
	CCGPs = hstack_test_model.get_all_CCGP(mnist_8, max_n_dichs=10, dichs_to_include={(0,1,2,3), (0,2,4,6)})
	assert True