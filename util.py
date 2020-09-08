import os
import sys
import numpy as np

def rodrigues(r):
	theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
	# avoid zero divide
	theta = np.maximum(theta, np.finfo(np.float64).tiny)
	r_hat = r / theta
	cos = np.cos(theta)
	z_stick = np.zeros(theta.shape[0])
	m = np.dstack([
	  z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
	  r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
	  -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
	).reshape([-1, 3, 3])
	i_cube = np.broadcast_to(
	  np.expand_dims(np.eye(3), axis=0),
	  [theta.shape[0], 3, 3]
	)
	A = np.transpose(r_hat, axes=[0, 2, 1])
	B = r_hat
	dot = np.matmul(A, B)
	R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
	return R
	
def model_summary(targets):
	print("")
	_print = lambda x: print('\t' + x)
	sep = '---------------------------'
	total = 0
	_print(sep)
	_print('MODEL SUMMARY')
	_print(sep)
	for t in targets:
		_print(t.name + '\t' + str(t.shape))
		total += np.prod(t.shape)
	_print(sep)
	_print('Total params: ' + str(total))
	_print(sep)
	print("")