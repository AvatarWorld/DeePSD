import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smpl.smpl_np import SMPLModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from Layers import *
from values import rest_pose

class DeePSD:
	# Builds models and initializes SMPL
	def __init__(self, checkpoint=None, rest_pose=rest_pose):
		# checkpoint: path to pre-trained model
		# with_body: will compute SMPL body vertices
		# rest_pose: SMPL rest pose for the dataset (star pose in CLOTH3D; A-pose in TailorNet; ...)
		self._build()
		
		smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
		self.SMPL = {
			0: SMPLModel(smpl_path + 'model_f.pkl', rest_pose),
			1: SMPLModel(smpl_path + 'model_m.pkl', rest_pose)
		}
		# load pre-trained
		if checkpoint is not None:
			print("Loading pre-trained model: " + checkpoint)
			self.load(checkpoint)

	# Builds model
	def _build(self):
		# Phi
		self._phi = [
			GraphConvolution((6, 32), act=tf.nn.relu, name='phi0'),
			GraphConvolution((32, 64), act=tf.nn.relu, name='phi1'),
			GraphConvolution((64, 128), act=tf.nn.relu, name='phi2'),
			GraphConvolution((128, 256), act=tf.nn.relu, name='phi3'),
		]
		# Omega
		self._omega = [
			GraphConvolution((256, 128), act=tf.nn.relu, name='omega0'),
			GraphConvolution((128, 64), act=tf.nn.relu, name='omega1'),
			GraphConvolution((64, 32), act=tf.nn.relu, name='omega2'),
			GraphConvolution((32, 24), act=tf.nn.relu, name='omega3')
		]
		# Psi
		self._psi = [
			FullyConnected((328, 256), act=tf.nn.relu, name='psi0'),
			FullyConnected((256, 128), act=tf.nn.relu, name='psi1'),
			FullyConnected((128, 64), act=tf.nn.relu, name='psi2'),
			FullyConnected((64, 3), name='psi3')
		]
	
	# Returns list of model variables
	def gather(self):
		vars = []
		for l in self._phi + self._omega + self._psi:
			vars += l.gather()
		return vars
	
	# loads pre-trained model
	def load(self, checkpoint):
		# checkpoint: path to pre-trained model
		# list vars
		vars = self.gather()
		# load vars values
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		values = np.load(checkpoint, allow_pickle=True)[()]
		# assign
		try: 
			for v in vars: v.assign(values[v.name])
		except: print("Mismatch between model and checkpoint")
		
	def save(self, checkpoint):
		# checkpoint: path to pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get vars values
		values = {v.name: v.numpy() for v in self.gather()}
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)
	
	# Computes geometric descriptors for each vertex of the template outfit
	def _descriptors(self, X, L):
		# X: template outfit verts
		# L: template outfit laplacian
		for l in self._phi: X = l(X, L)
		return X
	
	# Computes blend weights for each descriptor of the template outfit
	def _weights(self, X, L):
		# X: template outfit descriptors
		# L: template outfit laplacian
		for l in self._omega: X = l(X, L)
		# normalize weights to sum 1
		X = X / (tf.reduce_sum(X, axis=-1, keepdims=True) + 1e-7)
		return X
	
	# Computes deformations for each descriptor of the template outfit
	def _deformations(self, X, P, indices):
		# X: template outfit descriptors
		# P: poses
		# indices: splits among outfits (multiple outfits per batch)
		
		# tile
		P_tile = []
		for i in range(1, len(indices)):
			n = indices[i] - indices[i - 1]
			P_tile += [tf.tile(tf.expand_dims(P[i - 1], 0), [n, 1])]
		P_tile = tf.concat(P_tile, axis=0)
		P_tile = tf.cast(P_tile, tf.float32)
		# mlp
		X = tf.concat((X, P_tile), axis=-1)
		for l in self._psi:
			X = l(X)
		return X
	
	# Computes the skinning for each outfit/pose
	def _skinning(self, T, G, indices):
		V = []
		for i in range(1, len(indices)):
			s, e = indices[i - 1], indices[i]
			_T = T[s:e]
			_G = G[i - 1]
			_weights = self.W[s:e]
			_G = tf.einsum('ab,bcd->acd', _weights, _G)
			_T = tf.concat((_T, self._ones(tf.shape(_T))), axis=-1)
			_T = tf.linalg.matmul(_G, _T[:,:,None])[:,:3,0]
			V += [_T]
		return tf.concat(V, axis=0)
	
	# Computes the transformation matrices of each joint of the skeleton for each pose
	def _transforms(self, poses, shapes, genders, with_body):
		G = []
		B = []
		for p,s,g in zip(poses, shapes, genders):
			_G, _B = self.SMPL[g].set_params(pose=p, beta=s, with_body=with_body)
			G += [_G]
			B += [_B]
		return np.stack(G), np.stack(B)
	
	def _ones(self, T_shape):
		return tf.ones((T_shape[0], 1), tf.float32)
		
	def __call__(self, T, L, P, S, G, indices, with_body=False):
		# T: template outfits
		# L: laplacian
		# P: poses
		# S: shapes
		# G: genders
		# indices: splits among outfits (multiple outfits per batch)
		# with_body: compute posed SMPL body
		
		# surface descriptors
		X = self._descriptors(T, L)
		# weights
		self.W = self._weights(X, L)
		# deformations
		self.D = self._deformations(X, P, indices)
		
		# Compute forward kinematics and skinning
		Gs, B = self._transforms(P, S, G, with_body)
		V = self._skinning(T[:,:3] + self.D, Gs, indices)
		return V, B