import pdb
from tqdm import tqdm
import types

from itertools import combinations, permutations
from scipy.spatial.distance import cosine

import numpy as np
from numpy.linalg import norm
from numpy.random import shuffle, choice
from numpy.testing import assert_array_equal
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist

from data_tools import ImageDataset

# I'll temporarily store the following dichotomies in model
# Currently dichotomies are only binary
# CAUTION: always store the smallest elements of the dichotomy first
mnist_smallness = [range(0,4), range(4,8)]
mnist_parity = [list(map(lambda x: 2*x, range(4))), list(map(lambda x: 2*x + 1, range(4)))]
mnist_prod = [set(s1).intersection(set(s2)) for s2 in mnist_smallness for s1 in mnist_parity]

########################################################
## 3D annotation module ################################

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
	'''Annotate the point xyz with text s'''

	def __init__(self, s, xyz, *args, **kwargs):
		Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
		self._verts3d = xyz        

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.xy=(xs,ys)
		Annotation.draw(self, renderer)

def annotate3D(self, s, *args, **kwargs):
	'''add anotation text s to to Axes3d ax'''

	tag = Annotation3D(s, *args, **kwargs)
	self.add_artist(tag)

#########################################################
#########################################################


class Model:
	def __init__(self, kr_model):
		self.model = kr_model
		self.layers = [layer for layer in self.model.layers]
		self.n_layers = len(self.layers)

		self.model.compile(
			optimizer='rmsprop', #optimizer='sgd',
			loss='mean_squared_error',
			metrics=['accuracy']
		)

	def fit(self, ds, dich_name=None, epochs=10, batch_size=32, verbose=0):
		if dich_name is None:
			self.model.fit(ds.train['x'], ds.train['y_ohe'], epochs, batch_size, verbose=0)
		else:
			self.model.fit(ds.train['x'], ds.train['dichs'][dich_name], epochs, batch_size, verbose=0)

	def evaluate(self, ds, dich_name=None, batch_size=128):
		if dich_name is None:
			evl = self.model.evaluate(ds.test['x'], ds.test['y_ohe'], batch_size, verbose=0)
		else:
			evl = self.model.evaluate(ds.test['x'], ds.test['dichs'][dich_name], batch_size, verbose=0)
		return {'test_loss': evl[0], 'test_accuracy': evl[1]}

	def get_repr(self, ds, data, dimRed=None):
		"""
		dimRed: a dimensionality reduction instance (ex: sklearn.PCA instance), used for projecting high-dim data from the forward pass to 2d
		Caution, this instance must have a fit_transform method and a n_components attribute
		"""
		inp = self.model.input
		model_outputs = [layer.output for layer in self.layers]
		functors = [K.function([inp, K.learning_phase()], [out]) for out in model_outputs]

		# Currently the number of components for dimRed must be specified explicitly
		# n_components cannot (yet) be guessed from layer representations
		if dimRed is not None:
			dim = dimRed.n_components
		else:
			dim = None

		outs = [[] for layer in self.layers]
		expl_var_ratios = []
		expl_sing_vals = []

		# Extracting the representation from each layer
		for x in data['x']:
			dat = x[np.newaxis,...].reshape(1,ds.tot_dim)
			raw_out_list = [func([dat, 1]) for func in functors]
			for lay_id, layer in enumerate(self.layers):
				outs[lay_id].append(raw_out_list[lay_id][0].tolist()[0])
		outs[lay_id] = np.asarray(outs[lay_id])

		# Producing lower-dimensional dimensionality reduction
		rprs = [{
			'original': {
				'repr': [],
				'avg_lbl_repr': np.zeros((ds.n_labels, self.layers[lay_id].output_shape[1])),
			}
		} for lay_id in range(self.n_layers)]

		for lay_id in range(self.n_layers):
			rprs[lay_id]['original']['repr'] = outs[lay_id]
			for lbl_id, lbl in enumerate(ds.labels):
				filt = (data['y'] == lbl)
				rprs[lay_id]['original']['avg_lbl_repr'][lbl_id] = np.nanmean(np.array(rprs[lay_id]['original']['repr'])[filt,:], axis=0)

			if dim is not None:
				rprs[lay_id]['reduced'] = {
					'repr': [],
					'avg_lbl_repr': np.zeros((ds.n_labels, dim)),
					'expl_var': 0,
					'sing_vals': []
					}
				if (len(outs[lay_id][0]) > 1 and dim==2) or (len(outs[lay_id][0]) > 2 and dim==3):
					rprs[lay_id]['reduced']['repr'] = dimRed.fit_transform(outs[lay_id])
				else:
					rprs[lay_id]['reduced']['repr'] = np.array(outs[lay_id])
				rprs[lay_id]['reduced']['avg_lbl_repr'][lbl_id] = np.nanmean(np.array(rprs[lay_id]['reduced']['repr'])[filt,:], axis=0)

				rprs[lay_id]['reduced']['expl_var'] = dimRed.explained_variance_ratio_
				rprs[lay_id]['reduced']['sing_vals'] = dimRed.singular_values_

		return rprs

	def plot_reprs(self, ds, data, dimRed):

		# Get the representations using dimRed on our layer representations
		rprs = self.get_repr(ds, data, dimRed)

		if dimRed is not None:
			dim = dimRed.n_components
		else:
			dim = None

		# Creates subplots and unpacks the output array immediately
		fig = plt.figure(figsize=(8, 8*self.n_layers))
		axes = []
		gs1 = gridspec.GridSpec(self.n_layers,1)
		gs1.update(wspace=0.025, hspace=0.2) 
		
		vis_2d_lbl_noise = 0.1 * (0.5-np.random.rand(ds.n_labels, 2))
		vis_3d_lbl_noise = 0.1 * (0.5-np.random.rand(ds.n_labels, 3))

		for lay_id in range(self.n_layers):
			if dim==2 or len(rprs[lay_id]['original']['repr'][0])==2:
				ax = plt.subplot(gs1[lay_id])
				x = rprs[lay_id]['reduced']['repr'][:,0]
				y = rprs[lay_id]['reduced']['repr'][:,1]
				ax.scatter(x, y, color='red')
				x_range = np.absolute(ax.get_xlim()[1] - ax.get_xlim()[0])
				y_range = np.absolute(ax.get_ylim()[1] - ax.get_ylim()[0])
				for i, label in enumerate(ds.spl['y']):
					ax.annotate(label, (x[i]+vis_2d_lbl_noise[label,0]*x_range ,y[i]+vis_2d_lbl_noise[label,1]*y_range))
				axes.append(ax)
			elif dim==3 and len(rprs[lay_id]['original']['repr'][0])>2:
				ax = plt.subplot(gs1[lay_id], projection='3d')
				ax.annotate3D  = types.MethodType(annotate3D, ax)
				x = rprs[lay_id]['reduced']['repr'][:,0]
				y = rprs[lay_id]['reduced']['repr'][:,1]
				z = rprs[lay_id]['reduced']['repr'][:,2]
				xyz = [xyz_ for xyz_ in zip(x, y, z)]
				ax.scatter(x, y, z, c='red')
				for i, label in enumerate(ds.spl['y']):
					ax.annotate3D(s=label, xyz=xyz[i]+vis_3d_lbl_noise[label,:], fontsize=10, xytext=(-3,3),
				 textcoords='offset points', ha='right',va='bottom')
					axes.append(ax)

		return fig

	def sample_eval(self, ds, n_evals):
		fig = plt.figure(figsize=(8,8*n_evals))
		axes = []
		gs1 = gridspec.GridSpec(n_evals,1)
		gs1.update(wspace=0.025, hspace=0.6) 

		test_expl_ids = np.random.choice(ds.n_test, n_evals)

		for plt_id, expl_id in enumerate(test_expl_ids):
			ax = plt.subplot(gs1[plt_id])
			x = ds.test['x'][expl_id,:]
			y = ds.test['y'][expl_id]
			ax.imshow(
				x.reshape(ds.axes_dim)
			)
			pred = self.model.predict(x.reshape(1,ds.tot_dim))
			pred_is_even = (pred[0][0] > pred[0][1])
			pred_is_great = (pred[0][2] > pred[0][3])
			pred_text = "Class: {0:d}\nNumber is even: {1:s}\nNumber is smaller than 4: {2:s}".format(int(y), str(bool(pred_is_even)), str(bool(pred_is_great)))
			ax.text(0.5,-0.4, pred_text, size=12, ha="center", transform=ax.transAxes)
			axes.append(ax)

	def get_dich_PS(self, ds, lbl_repr, dich):  
		# To compute the PS for a layer representation, we have to find the max score accross all dichotomies
		dich_0 = dich
		dich_1 = np.setdiff1d(ds.labels, dich_0)
		PS = 0
		# Create a one-to-one pairing accross the dichotomy:
		for perm in permutations(range(len(dich_1))):
			# Compute all angles and average the results
			PS_perm = 0
			for i, rpr in enumerate(dich_0):
				vec_i = lbl_repr[rpr] - lbl_repr[dich_1[perm[i]]]
				vec_i = vec_i/norm(vec_i)
				for j in range(i+1, int(ds.n_labels/2)):
					vec_j = lbl_repr[dich_0[j]] - lbl_repr[dich_1[perm[j]]]
					vec_j = vec_j/norm(vec_j)
					PS_perm = PS_perm + np.dot(vec_i, vec_j)/6
			if PS_perm > PS:
				PS = PS_perm
		return PS

	def get_all_PS(self, ds, reds, lay_id):
		# To compute the PS for a layer representation, we have to find the max score accross all dichotomies
		assert ds.n_labels%2 == 0
		
		avg_lbl_repr = reds[lay_id]['original']['avg_lbl_repr']

		PS = {}
		# Let's loop over all dichotomies:
		dichs = list(combinations(ds.labels, int(ds.n_labels/2)))
		dichs = dichs[:int(len(dichs)/2)]
		for dich in tqdm(dichs):
			PS[dich] = self.get_dich_PS(ds, avg_lbl_repr, dich)
		PS = {k: v for k, v in sorted(PS.items(), key=lambda item: item[1], reverse=True)}

		return PS

	def get_dich_CCGP(self, ds, dich, n_labels_retained=2):
			"""
			CCGP is defined as the capacity to generalize on an unseen condition from training on a subset of possible conditions
			"""
			dich_0 = dich
			dich_1 = np.setdiff1d(ds.labels, dich_0)
			assert len(dich_0) == len(dich_1)

			# For now, labels will only be retained from dich_1 and will be random within dich_1
			retained_labels = np.random.choice(dich_1, size=n_labels_retained, replace=False)
			train_labels = np.setdiff1d(ds.labels, retained_labels)

			ds_gnr = ds.generate_gnr_set(train_labels)
			#ds_gnr.spl = ds.spl

			dich_name = ''.join([str(x) for x in dich_0])
			ds_gnr.build_dichLabels([dich_0, dich_1], dich_name)
			ds_gnr.train['dichs'][dich_name] = ds_gnr.train['dichs'][dich_name][:,0]
			ds_gnr.test['dichs'][dich_name] = ds_gnr.test['dichs'][dich_name][:,0]
			ds_gnr.gnr['dichs'][dich_name] = ds_gnr.gnr['dichs'][dich_name][:,0]
			
			# Create the linear classifier and train it on the submodel instance
			svc = LinearSVC()
			rpr = {
				'train': self.get_repr(ds_gnr, ds_gnr.train, dimRed=None),
				'test': self.get_repr(ds_gnr, ds_gnr.test, dimRed=None),
				'gnr': self.get_repr(ds_gnr, ds_gnr.gnr, dimRed=None)
			}

			CCGP_across_layers = []

			for lay_id in range(self.n_layers):
				#pca2 = PCA(n_components=2)
				#train_red_repr = pca2.fit_transform(rpr['train'][lay_id]['original']['repr'])
	            #svc.fit(train_red_repr, ds_gnr.train['dichs'][dich_name])
				svc.fit(rpr['train'][lay_id]['original']['repr'], ds_gnr.train['dichs'][dich_name])

				train_score = svc.score(rpr['train'][lay_id]['original']['repr'], ds_gnr.train['dichs'][dich_name])
				test_score = svc.score(rpr['test'][lay_id]['original']['repr'], ds_gnr.test['dichs'][dich_name])
				gnr_score = svc.score(rpr['gnr'][lay_id]['original']['repr'], ds_gnr.gnr['dichs'][dich_name])

				# Evaluate performance
				CCGP_across_layers.append({
					'train_labels': train_labels,
					'retained_labels': retained_labels,
					'train_score': train_score,
					'test_score': test_score,
					'gnr_score': gnr_score
				})

			return CCGP_across_layers

	def get_all_CCGP(self, ds, max_n_dichs=None, dichs_to_include=None):
		"""
		CCGP is defined as the capacity to generalize on an unseen condition from training on a subset of possible conditions
		"""
		assert ds.n_labels%2 == 0

		CCGP = {}
		all_dichs = list(combinations(ds.labels, int(ds.n_labels/2)))
		if max_n_dichs is not None:
			assert isinstance(max_n_dichs, int)
			if max_n_dichs < len(all_dichs):
				dich_set_ids = np.random.choice(range(len(all_dichs)), max_n_dichs, replace=False)
			else:
				dich_set_ids = range(len(all_dichs))
			dich_set = set([all_dichs[i] for i in dich_set_ids])
			if dichs_to_include is not None:
				dich_set = dich_set.union(set(dichs_to_include))
		else:
			dich_set = set(all_dichs)

		# Let's loop over all dichotomies:
		for dich in tqdm(dich_set):
			CCGP[dich] = self.get_dich_CCGP(ds, dich)

		return CCGP