import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import *
from values import *

samples = os.listdir(SRC)
for i,sample in enumerate(samples):
	print(str(i+1)+'/'+str(len(samples)))
	src = SRC + sample + '/'
	dst = SRC_PREPROCESS + sample + '/'
	for file in os.listdir(src):
		if not file.endswith('.obj'): continue
		fpath = src + file
		fout = dst + file.replace('.obj','_static.pc16')
		if os.path.isfile(fout): continue
		V, F, _, _ = readOBJ(fpath)
		writePC2(fout, V[None], True)