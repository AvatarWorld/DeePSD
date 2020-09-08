import os
import sys
import numpy as np
from math import cos, sin

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from values import *
from IO import *

def zRotMatrix(zrot):
	c, s = cos(zrot), sin(zrot)
	return np.array([[c, -s, 0],
					 [s,  c, 0],
					 [0,  0, 1]], np.float32)
					 
def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
	return np.array(out, np.int32)

samples = os.listdir(SRC)
for i,sample in enumerate(samples):
	print(str(i+1)+'/'+str(len(samples)))
	src = SRC + sample + '/'
	dst = SRC_PREPROCESS + sample + '/'
	# list garments
	garments = [f.replace('.obj','') for f in os.listdir(src) if f.endswith('.obj')]
	if not os.path.isdir(dst): os.mkdir(dst)
	with open(dst + 'outfit.txt', 'w') as f:
		for i,g in enumerate(garments):
			f.write(g + '\n')
	# merge rest garments into single outfit
	zrot = loadInfo(src + 'info.mat')['zrot']
	V, F = None, None
	for g in enumerate(garments):
		v, f, _, _ = readOBJ(src + garment + '.obj')
		f = quads2tris(f)
		if V is None:
			V = v
			F = f
		else:
			n = V.shape[0]
			V = np.concatenate((V, v), axis=0)
			F = np.concatenate((F, f + n), axis=0)
	V = (np.linalg.inv(zRotMatrix(zrot)) @ V.T).T
	writePC2(dst + 'rest.pc16', V[None], True)
	writeFaceBIN(dst + 'faces.bin', F)