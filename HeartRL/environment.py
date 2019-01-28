import numpy as np 
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_erosion
import malis
import random
from collections import namedtuple
import skimage.io as io
import os, sys, argparse, glob
from natsort import natsorted 
from img_aug_func import *
import copy
import math

def get_center (lbl_img):
	ske = skeletonize (lbl_img == 255)
	index_list = np.where (ske)
	index_mean = (np.mean (index_list[0]), np.mean (index_list[1]))
	index_zip = np.array (zip (index_list[0], index_list[1]))
	centroid_id = np.argmin (np.sum ((index_zip - index_mean) ** 2, axis=1))
	return index_zip [centroid_id]

def distance (p1, p2):
	return 1.0 * abs (p1[0] - p2[0]) +  abs (p1[1] - p2[1])

class State:
	def __init__ (self, _id, start, size, img_id, target, mov_dist):
		self.id = _id
		self.start = start
		self.size = size
		self.img_id = img_id
		self.target = target
		self.depth = 0
		self.mov_dist = mov_dist
		self.done = False
		self.acc_reward = 0
		self.action_his = []

	def center (self):
		return self.start[0] + self.size [0] / 2, self.start[1] + self.size [1] / 2

	def debug (self):
		print 'id:', self.id, 'start:', self.start, 'size:', self.size, 'img_id:', self.img_id, 'target:', self.target, 'depth:', self.depth

class Action:
	def __init__ (self, value):
		self.val = value

	def __len__ (self):
		return 5

	def numpy (self):
		ret = np.zeros (5, dtype=np.float32)
		ret [self.val] = 1
		return ret

class ActionSpace:
	def __init__ (self, length):
		self._len = length
		self.n = length

	def numpy (self, val):
		ret = np.zeros (self._len, dtype=np.float32)
		ret [self.val] = 1
		return ret

	def sample (self):
		return random.randint (0, self._len - 1) 

	def num_actions (self):
		return self.n

class ObservationSapce:
	def __init__ (self):
		pass

class Environment:
	def __init__ (self, raw_list, lbl_list):
		self.max_mov_dist = 80
		self.action_space = ActionSpace (5)

		self.raw_list = raw_list
		self.lbl_list = lbl_list

		self.reset ()
		
	def step (self, action):
		done = False
		reward = 0
		max_dist = 256 * 2
		max_reward = 1
		threshold_ratio = 0.95
		state = copy.deepcopy (self.state)
		self.set_state (state)
		# cur_dist =  distance (self.get_cen (), state.target)
		prev_state = copy.deepcopy (state)

		if state.done:
			done = True
			info = {
				'ale.lives': 0
			}
			reward = 0
			return self.observation (), reward, done, info


		action = Action (action)
		info = {
			'ale.lives': 1
		}

		if action.val == 4:
			info = {
				'ale.lives': 0
			}
			reward = (max_dist - distance (self.get_cen (), state.target)) / max_dist * max_reward
			if reward < max_reward * threshold_ratio:
				reward = 0
			state.done = True
			self.set_state (state)
			done = True
			return self.observation (), reward, done, info
		
		state.depth += 1
		state.id = state.id * 4 + action.val
		state.size = (state.size [0] / 2, state.size [1] / 2)
		state.action_his += [action.val]



		# if state.start [0] <= state.target [0] < state.start [0] + state.size [0] \
		# 	and state.start[1] <= state.target [1] < state.start [1] + state.size [1] \
		# 	and state.depth == 1:
		# 	reward += 5

		# reward += state.acc_reward 

		self.set_state (state)
		if state.done:
			reward = (max_dist - distance (self.get_cen (), state.target)) / max_dist * max_reward
		if reward < max_reward * threshold_ratio:
			reward = 0
		return self.observation (), reward, done, info


	def sample_action (self):
		return random.randint (0, 4)

	def get_cen (self):
		return self.state.center ()

	def reset (self):
		img_id = np.random.randint (len (self.raw_list))
		self.raw = self.raw_list [img_id]
		self.lbl = self.lbl_list [img_id]
		self.target = get_center (self.lbl)
		max_mov_dist = self.max_mov_dist
		mov_dist = (np.random.randint (-max_mov_dist, max_mov_dist + 1), 
			np.random.randint (-max_mov_dist, max_mov_dist + 1))
		self.target [0] -= mov_dist[0]; self.target[1] -= mov_dist[1]
		self.state = State (0, [0, 0], self.raw.shape, img_id, self.target, mov_dist)
		self.cur_dist = distance (self.get_cen (), self.target)
		return self.observation ()

	def get_state (self):
		return copy.deepcopy (self.state)

	def set_state (self, state):
		img_id = state.img_id
		self.raw = self.raw_list [img_id]
		self.lbl = self.lbl_list [img_id]
		self.target = state.target
		self.state = copy.deepcopy (state)
		self.cur_dist = distance (self.get_cen (), self.target)

	def get_mask (self, action_his=None, state=None):
		
		mask_list = []

		if action_his is not None:
			size = [256, 256]
			start = [0, 0]
			for act in action_his:
				size [0] /= 2; size[1] /= 2;

				if act == 0:
					start [0] += 0
					start [1] += 0

				if act == 1:
					start [0] += 0
					start [1] += size [1]

				if act == 2:
					start [0] += size [0]
					start [1] += 0

				if act == 3:
					start [0] += size [0]
					start [1] += size [1]

				mask = np.zeros_like (self.raw)
				mask [start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = 255
				# print 'debug ', mask [start[0]:start[0]+size[0], start[1]:start[1]+size[1]].shape

				mask_list += [mask]

			# print len (mask_list)

			while (len (mask_list) != 4):
				mask_list += [np.zeros_like (self.raw)]

			return mask_list

		
		mask = np.zeros_like (self.raw)

		if state == None:
			size = self.state.size
			start = self.state.start
		else:
			size = state.size
			start = state.start

		mask [start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = 255
		return mask

	def observation (self):
		'''
		Observation of size (W, H, 2)
		'''
		raw = self.raw [::]
		raw = mov_img (raw, self.state.mov_dist, self.max_mov_dist)
		raw = np.expand_dims (raw, -1)
		mask_list = self.get_mask (action_his=self.state.action_his)
		for i in range (len (mask_list)):
			mask_list[i] = np.expand_dims (mask_list[i], -1)
		current_iteration = len (self.state.action_his) - 1;
		if (current_iteration < 0):
			current_iteration = 0
		ret = np.concatenate ([raw] + [mask_list [current_iteration]], -1)
		ret = ret.astype (np.uint8)
		return ret

	def get_boundary_mask (self, img, pad=5):

		size = self.state.size
		start = copy.deepcopy (self.state.start)

		start [0] += pad
		start [1] += pad

		if size [0] < 10:
			return img
			
		img [:,start[0]:start[0]+size[0], start[1],:] = 0
		img [:,start[0], start[1]:start[1]+size[1],:] = 0
		img [:,start[0]:start[0]+size[0], start[1]+size[1]-1,:] = 0
		img [:,start[0]+size[0]-1, start[1]:start[1]+size[1],:] = 0

		img [:,start[0]:start[0]+size[0], start[1],:] = 255
		img [:,start[0], start[1]:start[1]+size[1],:] = 255
		img [:,start[0]:start[0]+size[0], start[1]+size[1]-1,:] = 255
		img [:,start[0]+size[0]-1, start[1]:start[1]+size[1],:] = 255
		return img

	def get_action_space (self):
		return self.action_space

	def get_log_img (self):
		log_img_pad = 5
		obs = self.observation ()
		log_img = obs [...,0]
		log_img = np.pad (log_img, pad_width=log_img_pad, mode='constant', constant_values=255)
		log_img = np.repeat (np.expand_dims (log_img, -1), 3, -1)
		log_img = np.expand_dims (log_img, 0)
		log_img = self.get_boundary_mask (log_img, log_img_pad).astype (np.uint8)
		target = self.get_state ().target

		# Draw center, target
		dxs = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
		dys = [1, 0, -1, 1, 0, -1, 1, 0, -1]

		center = self.get_cen ()

		for d in range (len (dxs)):
			log_img [:, target[0] + dxs[d] + log_img_pad, target[1] + dys[d] + log_img_pad] = np.array ([255, 0, 0], dtype=np.uint8)
			clamp = lambda u: max (0, min (u, 255))
			cx = clamp (center[0] + dxs[d] + log_img_pad); cy = clamp (center[1] + dys[d] + log_img_pad);
			log_img [:, cx, cy] = np.array ([0, 255, 0], dtype=np.uint8)

		log_img = np.squeeze (log_img)

		# log_masks = []
		# for c in range (1, 5):
		# 	log_mask = obs [...,c]
		# 	log_mask = np.pad (log_mask, pad_width=log_img_pad, mode='constant', constant_values=255)
		# 	log_mask = np.repeat (np.expand_dims (log_mask, -1), 3, -1)
		# 	log_masks +=  [log_mask]
		# log_img = np.concatenate ([log_img] + log_masks, 0)

		return log_img 





