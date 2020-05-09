import json, os
import numpy as np
import matplotlib.pyplot as plt
# pour plt.imshow(x_train[1])
import functools, operator
from smtlib import smtlib

tfx = False
if tfx:
	import tensorflow as tf

prod = lambda l: functools.reduce(operator.mul,l,1)
assert prod(range(1,4)) == 6

def activ_relu(v):
	""
	with np.nditer(v, op_flags=['readwrite']) as it:
		for x in it:
			if x < 0.0: x[...] = 0.0

def activ_softmax(v):
	""
	assert False

def predict(nn_js, samples, out=None):
	""
	slow = False
	configuration, weights = nn_js
	# compil
	weights = [np.array(wx, np.float32) for wx in weights]
	regs = [None]
	sh = samples[0].shape
	nxt_w_i = 0
	for l_i, conf_layer in enumerate(configuration['layers']):
		kind = conf_layer['class_name']
		if kind == 'Dense':
			assert conf_layer['config']['dtype'] == 'float32'
			assert conf_layer['config']['use_bias']
			units = conf_layer['config']['units']
			w1,w2 = weights[nxt_w_i:nxt_w_i+2]
			nxt_w_i += 2
			sh1 = w1.shape; sh2 = w2.shape
			assert sh == sh1[:-1] and sh1[-1:] == sh2 == (units,)
			regs.append(np.zeros(sh2, dtype=np.float32))
			sh = sh2
		elif kind == 'Dropout':
			pass
		elif kind == 'Flatten':
			d1,d2 = sh
			assert False
		elif kind == 'InputLayer':
			assert conf_layer['config']['batch_input_shape'] == [None] + list(sh), conf_layer['config']
			assert conf_layer['config']['dtype'] == 'float32'
		else:
			assert False, kind
	# sh est le shape de sortie
	if out is None:
		out = np.zeros((samples.shape[0],)+sh, dtype=np.float32)
	else:
		assert out.shape == (samples.shape[0],)+sh and out.dtype == np.float32
	#
	p_l = []
	for sample, result in zip(samples,out):
		regs[0] = sample; regs[-1] = result; r_i = 0
		neurons = sample
		nxt_w_i = 0
		for l_i, conf_layer in enumerate(configuration['layers']):
			assert isinstance(neurons, np.ndarray) # and len(neurons.shape) == 1
			kind = conf_layer['class_name']
			if kind == 'Dense':
				w1_,w2_ = weights[nxt_w_i:nxt_w_i+2]
				#w1 = np.array(w1_, np.float32); w2 = np.array(w2_, np.float32);
				w1 = w1_ ; w2 = w2_
				nxt_w_i += 2
				np.matmul(regs[r_i], w1, out=regs[r_i+1]); r_i +=1
				if slow:
					sh1 = w1.shape
					assert neurons.shape == sh1[:1]
					# 'a' = np, 'b'= tf
					neurons1a = np.matmul(neurons,w1)
					assert all(regs[r_i] == neurons1a)
				if tfx:
					neurons1b = tf.matmul(neurons.reshape((1,len(neurons))),w1)
					assert all(neurons1a == neurons1b[0].numpy())
				regs[r_i] += w2
				if slow:
					neurons1a += w2
					assert all(regs[r_i] == neurons1a)
				if tfx:
					neurons1b = tf.nn.bias_add(neurons1b, w2)
					assert all(neurons1a == neurons1b[0].numpy())
				activ = conf_layer['config']['activation']
				if activ == 'relu':
					activ_relu(regs[r_i])
					if slow:
						neurons1a = np.fromiter(((0 if x < 0 else x) for x in neurons1a), np.float32)
						assert all(regs[r_i] == neurons1a)
					if tfx:
						neurons1b = tf.nn.relu(neurons1b)
						assert all(neurons1a == neurons1b[0].numpy())
				elif activ == 'softmax':
					activ_softmax(regs[r_i])
					if slow:
						neurons1a = np.exp(neurons1a - np.max(neurons1a))
						neurons1a /= np.sum(neurons1a)
						assert all(regs[r_i] == neurons1a)
					if tfx:
						neurons1b = tf.nn.softmax(neurons1b)
						errs = np.abs(neurons1a - neurons1b[0].numpy()) / (neurons1a + neurons1b[0].numpy())
						if np.max(errs) >= 1e-6:
							print('???? 1a - 1b max rel err : {}'.format(np.max(errs)))
				elif activ == 'linear':
					_ = 2+2
				else:
					assert False, activ
				if slow: neurons = neurons1a
			elif kind == 'Dropout':
				_ = 2+2
			elif kind == 'Flatten':
				d1,d2 = neurons.shape
				neurons = neurons.reshape((d1*d2,))
			elif kind == 'InputLayer':
				_ = 2+2
			else:
				assert False, kind
			if slow: assert isinstance(neurons, np.ndarray) and len(neurons.shape) == 1
		#assert nxt_b_i == len(buf_l)
		assert nxt_w_i == len(weights)
		if slow:
			p_l.append(neurons)
	if slow:
		r = np.array(p_l)
		assert (r == out).all()
	return out

def approx_float(w, levels):
	""
	return np.floor(w*levels+0.5)/levels

def approx(nn_js, levels):
	""
	def app_wx(wx, levels):
		"wx est soit une liste de floats, soit une liste de listes de floats"
		if isinstance(wx[0], list):
			return [app_wx(wy,levels) for wy in wx]
		else:
			return [approx_float(w,levels) for w in wx]
	#
	configuration, weights = nn_js
	return configuration, [app_wx(wx, levels) for wx in weights]

def quant(nn_js, levels, max_denom=None):
	""
	def quant_wx(wx, levels):
		"wx est soit une liste de floats, soit une liste de listes de floats"
		if isinstance(wx[0], list):
			return [quant_wx(wy,levels) for wy in wx]
		else:
			return [operator.truediv(*smtlib.float2rat(w,levels,max_denom)) for w in wx]
	#
	configuration, weights = nn_js
	return configuration, [quant_wx(wx, levels) for wx in weights]

def max_arr(a):
	""
	return max(abs(a.min()),abs(a.max()))

def normalize(nn_js, levels=None):
	""
	def round_arr(a, levels):
		""
		return np.floor(a*levels+0.5)
	conf, weights = nn_js
	assert len(weights) == 4
	npw = [np.array(wx, np.float32) for wx in weights]
	m = [max_arr(a) for a in npw]
	m1 = max(m[:2])
	one = 1-1/1024
	npw[0] *= one/m1 ; npw[1] *= one/m1
	m2 = max(m[2:])
	npw[2] *= one/m2 ; npw[3] *= one/m2
	if levels:
		npw = [round_arr(a,levels) for a in npw]
	return conf, [w.tolist() for w in npw]

x_train0, y_train, x_test0, y_test = [None]*4

def perfo(nn_js):
	""
	sz = x_train0.shape[1]
	pred = predict(nn_js, x_train0.reshape(60000, sz*sz).astype('float32'))
	estim = pred.argmax(axis=1)
	print('errs : {}'.format(sum(estim != y_train)))
	#
	pred = predict(nn_js, x_test0.reshape(10000, sz*sz).astype('float32'))
	estim = pred.argmax(axis=1)
	print('errs : {}'.format(sum(estim != y_test)))

if __name__ == "__main__":
	sz = 12
	levels = 5
	name = '2_layer_mlp_SMALL{}'.format(sz) if sz else '2_layer_mlp'
	fd = open(name+'.json', encoding='utf8')
	nn_js = json.load(fd)
	fd.close()
	#
	datafile = 'mnist_small{}.npz'.format(sz) if sz else 'mnist.npz'
	npz = np.load(datafile)
	x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
	npz.close()
	#x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	#
	perfo(nn_js)
	x_train0 >>= 3
	x_test0 >>= 3
	nn_js_n = normalize(nn_js, (1<<levels)-0.51) # 7.49 optimal pour full,  31.49 pour small12 et small14 
	perfo(nn_js_n)
	#
	fd = open('{}_Q{}.json'.format(name,levels), 'w')
	json.dump(nn_js_n,fd)
	fd.close()
