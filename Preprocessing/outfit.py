import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from values import *

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