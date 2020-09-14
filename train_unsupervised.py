import os
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.DeePSD import DeePSD

from util import model_summary
from losses import *

""" ARGS """
# gpu_id: GPU slot to run model
# name: name under which model checkpoints will be saved
# checkpoint: pre-trained model (must be in ./checkpoints/ folder)
gpu_id = sys.argv[1] # mandatory
name = sys.argv[2]   # mandatory
checkpoint = None
if len(sys.argv) > 3:
	checkpoint = 'checkpoints/' + sys.argv[3]

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" TRAIN PARAMS """
# use of 'virtual_batch' due to VRAM limitations
batch_size = 4
virtual_batch_size = 1 if checkpoint is None else 4 # if fine-tuning, bigger batch
num_epochs = 10 if checkpoint is None else 4 # if fine-tuning, fewer epochs

""" MODEL """
print("Building model...")
model = DeePSD(checkpoint)
tgts = model.gather() # model weights
model_summary(tgts)
optimizer = tf.optimizers.Adam()

""" DATA """
print("Reading data...")
tr_txt = 'Data/train.txt'
val_txt = 'Data/val.txt'
tr_data = Data(tr_txt, batch_size=batch_size)
val_data = Data(val_txt, batch_size=batch_size)

tr_steps = floor(len(tr_data._samples) / batch_size)
val_steps = floor(len(val_data._samples) / batch_size)
for epoch in range(num_epochs):
	if (epoch + 1) % 2 == 0: virtual_batch_size *= 2
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	metrics = [0] * 4 # L2-norm, Edge, Bend, Collision
	cgrds = None
	start = time()
	for step in range(tr_steps):
		""" I/O """
		batch = tr_data.next()
		""" Train step """
		with tf.GradientTape() as tape:
			pred, body = model(
						batch['template'],
						batch['laplacians'],
						batch['poses'],
						batch['shapes'],
						batch['genders'],
						batch['indices'],
						with_body=True
					)					
			# Losses & Metrics
			_, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
			L_edge, E_edge = edge_loss(pred, batch['template'][:,:3], batch['edges'])
			L_bend, E_bend = bend_loss(pred, batch['faces'], batch['laplacians'])
			L_collision, E_collision = collision_loss(pred, body, model.SMPL[0].faces, batch['indices'])
			loss = L_edge + .005 * L_bend + L_collision
			# unsupervised prior
			L_balance = 10 ** ((epoch - 4) / 2)
			L_prior = tf.reduce_sum((model.W - batch['weights_prior'])**2)
			L_prior += weights_smoothness(model.W, batch['laplacians'])
			L_prior += tf.reduce_sum(model.D ** 2)
			loss = L_balance * loss + L_prior
		""" Backprop """
		grads = tape.gradient(loss, tgts)
		# Few batches and sum of gradients (for insufficient VRAM)
		if virtual_batch_size is not None:
			if cgrds is None: cgrds = grads
			else: cgrds = [c + g for c,g in zip(cgrds,grads)]
			if (step + 1) % virtual_batch_size == 0:
				optimizer.apply_gradients(zip(cgrds, tgts))
				cgrds = None
		else:
			optimizer.apply_gradients(zip(grads, tgts))
		""" Progress """
		metrics[0] += E_L2.numpy()
		metrics[1] += E_edge.numpy()
		metrics[2] += E_bend.numpy()
		metrics[3] += E_collision
		total_time = time() - start
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % 100 == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * metrics[0] / (1+step)) 
					+ ' - '
					+ 'E: {:.2f}'.format(1000 * metrics[1] / (1+step))
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[2] / (1+step))
					+ ' - '
					+ 'C: {:.4f}'.format(metrics[3] / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	metrics = [m / (step + 1) for m in metrics]
	print("")
	print("Total error: {:.5f}".format(1000 * metrics[0]))
	print("Total edge: {:.5f}".format(1000 * metrics[1]))
	print("Total bending: {:.5f}".format(metrics[2]))
	print("Total collision: {:.5f}".format(metrics[3]))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" VALIDATION """
	print("Validating...")
	total_time = 0
	metrics = [0] * 4 # L2-norm, Edge, Bend, Collision
	start = time()
	for step in range(val_steps):
		""" I/O """
		batch = val_data.next()
		""" Forward pass """
		pred, body = model(
					batch['template'],
					batch['laplacians'],
					batch['poses'],
					batch['shapes'],
					batch['genders'],
					batch['indices'],
					with_body=True
				)
		""" Metrics """
		_, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
		_, E_edge = edge_loss(pred, batch['template'][:,:3], batch['edges'])
		_, E_bend = bend_loss(pred, batch['faces'], batch['laplacians'])
		_, E_collision = collision_loss(pred, body, model.SMPL[0].faces, batch['indices'])
		""" Progress """
		metrics[0] += E_L2.numpy()
		metrics[1] += E_edge.numpy()
		metrics[2] += E_bend.numpy()
		metrics[3] += E_collision
		total_time = time() - start
		ETA = (val_steps - step - 1) * (total_time / (1+step))	
		if (step + 1) % 10 == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(val_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * metrics[0] / (1+step)) 
					+ ' - '
					+ 'E: {:.2f}'.format(1000 * metrics[1] / (1+step))
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[2] / (1+step))
					+ ' - '
					+ 'C: {:.4f}'.format(metrics[3] / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	metrics = [m / (step + 1) for m in metrics]
	print("")
	print("Total error: {:.5f}".format(1000 * metrics[0]))
	print("Total edge: {:.5f}".format(1000 * metrics[1]))
	print("Total bending: {:.5f}".format(metrics[2]))
	print("Total collision: {:.5f}".format(metrics[3]))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" Save checkpoint """
	model.save('checkpoints/' + name + str(epoch))
