import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from scipy.spatial import cKDTree

def L2_loss(x, y, indices):
	# x: predicted outfits
	# y: ground truth
	# indices: to split batch into single outfits
	D = tf.reduce_sum((x - y) ** 2, -1)
	loss = tf.reduce_sum(D)
	err = []
	for i in range(1, len(indices)):
		s, e = indices[i - 1], indices[i]
		err += [tf.reduce_mean(tf.sqrt(D[s:e]))]
	err = tf.reduce_mean(err)
	return loss, err

def edge_loss(x, y, E):
	# x: predicted outfits
	# y: template outfits
	# E: Nx2 array of edges
	x_e = tf.gather(x, E[:,0], axis=0) - tf.gather(x, E[:,1], axis=0)
	x_e = tf.reduce_sum(x_e ** 2, -1)
	x_e = tf.sqrt(x_e)
	y_e = tf.gather(y, E[:,0], axis=0) - tf.gather(y, E[:,1], axis=0)
	y_e = tf.reduce_sum(y_e ** 2, -1)
	y_e = tf.sqrt(y_e)
	d_e = tf.abs(x_e - y_e)
	loss = tf.reduce_sum(d_e)
	err = tf.reduce_mean(d_e)
	return loss, err

def bend_loss(x, F, L):
	# x: predicted outfits
	# F: faces
	# L: laplacian
	VN = tfg.geometry.representation.mesh.normals.vertex_normals(x, F)
	bend = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
		VN, L, sizes=None,
		edge_function=lambda x,y: (x - y)**2,
		reduction='weighted',
		edge_function_kwargs={}
	)
	bend_dist = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
		VN, L, sizes=None,
		edge_function=lambda x,y: 1 - tf.einsum('ab,ab->a', x, y)[:,None],
		reduction='weighted',
		edge_function_kwargs={}
	)
	bend_dist = tf.clip_by_value(bend_dist, 0, 2)
	return tf.reduce_sum(bend), tf.reduce_mean(bend_dist)
	
def collision_loss(x, B, B_F, indices, thr=.005):
	# x: predicted outfits
	# B: posed human bodies
	# B_F: human body faces
	# indices: to split batch into single outfits
	# thr: collision threshold
	loss = 0
	vcount = []
	for i in range(1, len(indices)):
		s, e = indices[i - 1], indices[i]
		_x = x[s:e]
		_B = np.float32(B[i - 1])
		# build body KDTree
		tree = cKDTree(_B)
		_, idx = tree.query(_x.numpy())
		# to nearest
		D = _x - _B[idx]
		# body normals
		B_vn = tfg.geometry.representation.mesh.normals.vertex_normals(_B, B_F)
		# corresponding normals
		VN = tf.gather(B_vn, idx, axis=0)
		# dot product
		dot = tf.einsum('ab,ab->a', D, VN)
		vcount += [tf.reduce_sum(tf.cast(tf.math.less(dot, 0), tf.float32)) / _x.shape[0]]
		# collision if dot < 0  --> -dot > 0
		dot = -tf.minimum(dot, thr)
		loss += tf.reduce_sum(dot)
	return tf.reduce_sum(dot), np.array(vcount).mean()
	
def weights_smoothness(W, L):
	# W: predicted weights
	# L: laplacian
	return tf.reduce_sum(tfg.geometry.convolution.graph_convolution.edge_convolution_template(
		W, L, sizes=None,
		edge_function=lambda x,y: tf.abs(x - y),
		reduction='weighted',
		edge_function_kwargs={}
	))