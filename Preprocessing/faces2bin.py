import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import readOBJ, writeFaceBIN
from values import *

def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
	return np.array(out, np.int32)

samples = os.listdir(SRC)
N = len(samples)
for i,sample in enumerate(samples):
	print("Sample " + str(i+1) + '/' + str(N))
	src = SRC + sample + '/'
	dst = SRC_PREPROCESS + sample + '/'
	if not os.path.isdir(dst): os.mkdir(dst)
	for file in os.listdir(src):
		if not file.endswith('.obj'): continue
		if os.path.isfile(dst + file.replace('.obj','.bin')): continue
		_, F, _, _ = readOBJ(src + file)
		F = quads2tris(F)
		writeFaceBIN(dst + file.replace('.obj','.bin'), F)