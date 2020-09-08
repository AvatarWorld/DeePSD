import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *

samples = os.listdir(SRC_PREPROCESS)
N = len(samples)

for i,sample in enumerate(samples):
	print("Sample " + str(i+1) + '/' + str(N))
	file = SRC_PREPROCESS + sample + '/' + 'faces.bin'
	F = readFaceBIN(file)
	E = faces2edges(F)
	E = np.array(list(E))
	file = file.replace('faces','edges')
	writeEdgeBIN(file, E)