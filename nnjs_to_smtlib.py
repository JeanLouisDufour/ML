levels = 255  ## 1.0 == levels -> quantum = 1/levels
levels = None # non quantifie

import json, numpy as np
from smtlib import smtlib
#
import functools, operator
prod = lambda l: functools.reduce(operator.mul,l,1)

def float2int(f):
	""
	return int(np.floor(f*levels+0.5))

defineArray_d = {}

def defineArray(smtobj, name, a):
	""
	if not isinstance(a,np.ndarray):
		a = np.array(a)
	sh = a.shape
	assert len(sh) in (1,2)
	if len(sh) == 2:
		a = a.reshape((prod(sh),))
	for i,ai in enumerate(a):
		if levels:
			smtobj.defineFun(name+str(i),[],'Int',float2int(ai))
		else:
			smtobj.defineFun(name+str(i),[],'Real',float(ai))
	defineArray_d[name] = len(a)
	
def defineArrayNoisyAdd(smtobj, name, a):
	""
	assert levels is None
	if not isinstance(a,np.ndarray):
		a = np.array(a)
	sh = a.shape
	assert len(sh) in (1,2)
	if len(sh) == 2:
		a = a.reshape((prod(sh),))
	for i,ai in enumerate(a):
		n = 'noise'+str(i)
		e = ['+',float(ai),n] if ai < 0.5 else ['-',float(ai),n]
		smtobj.defineFun(name+str(i),[],'Real',e)
	defineArray_d[name] = len(a)
	
def defineArrayNoisyIf(smtobj, name, a):
	""
	if not isinstance(a,np.ndarray):
		a = np.array(a)
	sh = a.shape
	assert len(sh) in (1,2)
	if len(sh) == 2:
		a = a.reshape((prod(sh),))
	for i,ai in enumerate(a):
		n = 'noise'+str(i)
		if levels:
			e = ['ite',['>',n,0], n, float2int(ai)]
			smtobj.defineFun(name+str(i),[],'Int',e)
		else:
			e = ['+',float(ai),n] if ai < 0.5 else ['-',float(ai),n]
			e = ['ite',['>',n,0.0], n, float(ai)]
			smtobj.defineFun(name+str(i),[],'Real',e)
	defineArray_d[name] = len(a)

def defineArrayLinear(smtobj, prefix_out, prefix_in, A, B):
	""
	sz_in = defineArray_d[prefix_in]
	if not isinstance(A,np.ndarray):
		A = np.array(A)
	assert len(A.shape) == 2
	assert A.shape[0] == sz_in
	if not isinstance(B,np.ndarray):
		B = np.array(B)
	assert A.shape[1:] == B.shape
	for i,Bi in enumerate(B):
		se = [['*', prefix_in+str(j), (float2int(Aji) if levels else float(Aji))] for j,Aji in enumerate(A[:,i])]
		se = ['+', (float2int(Bi) if levels else float(Bi))] + se
		smtobj.defineFun(prefix_out+str(i),[],('Int' if levels else 'Real'),se)
	defineArray_d[prefix_out] = len(B)

def defineArrayReLU(smtobj, prefix_out, prefix_in, sz):
	""
	sz_in = defineArray_d[prefix_in]
	assert sz_in == sz
	for i in range(sz):
		inp = prefix_in+str(i)
		if levels:
			smtobj.defineFun(prefix_out+str(i),[],'Int',['ite', ['<',inp,0], 0, inp])
		else:
			smtobj.defineFun(prefix_out+str(i),[],'Real',['ite', ['<',inp,0.0], 0.0, inp])
	defineArray_d[prefix_out] = sz

def defineArraySoftmax(smtobj, prefix_out, prefix_in, sz):
	""
	defineArrayReLU(smtobj, prefix_out, prefix_in, sz)

def printArray(smtobj, prefix, sz):
	""
	sz_in = defineArray_d[prefix]
	assert sz_in == sz
	_ = 2+2

def defineMaxIsHere(smtobj, var, prefix, here):
	""
	assert levels is None
	sz = defineArray_d[prefix]
	assert 0 <= here < sz
	theMax = prefix+str(here)
	b = ['and']
	for i in range(sz):
		if i == here: continue
		b.append(['>', theMax, prefix+str(i)])
	smtobj.defineFun(var,[],'Bool', b)	

def defineArgMax(smtobj, prefix_in):
	""
	sz = defineArray_d[prefix_in]
	prefix_out = prefix_in + '_ArgMax_is_'
	for here in range(sz):
		theMax = prefix_in+str(here)
		b = ['and']
		for i in range(sz):
			if i == here: continue
			b.append(['>=', theMax, ['+',prefix_in+str(i),'offset']])
		if levels:
			smtobj.defineFun(prefix_out+str(here),[['offset','Int']],'Bool', b)
		else:
			smtobj.defineFun(prefix_out+str(here),[['offset','Real']],'Bool', b)
	defineArray_d[prefix_out] = sz

def defineNoise(smtobj, sz, i_max, s_max, nb_max=None):
	""
	assert isinstance(i_max,float) and isinstance(s_max,float)
	if nb_max is None:
		nb_max = sz
	prefix = 'noise'
	if levels:
		smtobj.defineFun(prefix+'_i_max',[],'Int', float2int(i_max))
		smtobj.defineFun(prefix+'_s_max',[],'Int', float2int(s_max))
	else:
		smtobj.defineFun(prefix+'_i_max',[],'Real', i_max)
		smtobj.defineFun(prefix+'_s_max',[],'Real', s_max)
	smtobj.defineFun(prefix+'_nb_max',[],'Int', nb_max)
	s = ['+']
	nb = ['+']
	for i in range(sz):
		name = prefix+str(i)
		s.append(name)
		if levels:
			nb.append(['ite',['>',name,0],1,0])
			smtobj.declareConst(name, 'Int')
			smtobj.Assert(['and',['>=',name,0],['<=',name,prefix+'_i_max']])
		else:
			nb.append(['ite',['>',name,0.0],1,0])
			smtobj.declareConst(name, 'Real')
			smtobj.Assert(['and',['>=',name,0.0],['<=',name,prefix+'_i_max']])
	if levels:
		smtobj.defineFun(prefix+'_s',[],'Int', s)
	else:
		smtobj.defineFun(prefix+'_s',[],'Real', s)
	smtobj.Assert(['<=',prefix+'_s',prefix+'_s_max'])
	smtobj.defineFun(prefix+'_nb',[],'Int', nb)
	smtobj.Assert(['<=',prefix+'_nb',prefix+'_nb_max'])
	defineArray_d[prefix] = sz

def translate(nn_js, samples, results=None):
	"""
	
	"""
	smtobj = smtlib('QF_LIRA')
	smtobj.setOption(':produce-models','true')
	smtobj.float_mantissa = 64
	configuration, weights = nn_js
	# compil
	if False:
		defineNoise(smtobj, prod(samples.shape[1:]), 0.1, 10.0)
	else:
		defineNoise(smtobj, prod(samples.shape[1:]), 1.0, 100.0)
	#
	p_l = []
	for s_i, sample in enumerate(samples):
		prefix = 's'+str(s_i)
		neurons = sample
		if False:
			defineArray(smtobj, prefix+'n0_', neurons)
		elif False:
			defineArrayNoisyAdd(smtobj, prefix+'n0_', neurons)
		else:
			defineArrayNoisyIf(smtobj, prefix+'n0_', neurons)
		nxt_n_i = 0
		nxt_w_i = 0
		for l_i, conf_layer in enumerate(configuration['layers']):
			assert isinstance(neurons, np.ndarray) # and len(neurons.shape) == 1
			kind = conf_layer['class_name']
			config = conf_layer['config']
			if kind == 'Dense':
				w1_,w2_ = weights[nxt_w_i:nxt_w_i+2]
				w1 = np.array(w1_, np.float32); w2 = np.array(w2_, np.float32); 
				nxt_w_i += 2
				sh1 = w1.shape; sh2 = w2.shape;
				assert sh1[-1:] == sh2
				assert neurons.shape == sh1[:len(neurons.shape)]
				neurons1a = np.matmul(neurons,w1) + w2
				defineArrayLinear(smtobj, \
					  prefix+'n{}_'.format(nxt_n_i+1), \
					  prefix+'n{}_'.format(nxt_n_i), w1, w2)
				nxt_n_i += 1
				activ = config['activation']
				if activ == 'relu':
					neurons1a = np.fromiter(((0 if x < 0 else x) for x in neurons1a), np.float32)
					defineArrayReLU(smtobj, \
						 prefix+'n{}_'.format(nxt_n_i+1), \
						 prefix+'n{}_'.format(nxt_n_i), len(neurons1a))
					nxt_n_i += 1
				elif activ == 'softmax':
					neurons1a = np.exp(neurons1a)
					# neurons1a *= (np.float32(1)/ np.sum(neurons1a))
					neurons1a *= (np.float64(1)/ np.sum(neurons1a.astype(np.float64)))
					defineArraySoftmax(smtobj, \
						 prefix+'n{}_'.format(nxt_n_i+1), \
						 prefix+'n{}_'.format(nxt_n_i), len(neurons1a))
					nxt_n_i += 1
				else:
					assert activ == 'linear'
				neurons = neurons1a
			elif kind == 'Dropout':
				_ = 2+2
			elif kind == 'Flatten':
				assert l_i == 0
				d1,d2 = neurons.shape
				neurons = neurons.reshape((d1*d2,))
			elif kind == 'InputLayer':
				assert l_i == 0
				sh = config['batch_input_shape']
				assert prod(neurons.shape) == prod(sh[1:])
				if len(neurons.shape) >= 2:
					d1,d2 = neurons.shape
					neurons = neurons.reshape((d1*d2,))
			else:
				assert False, kind
			assert isinstance(neurons, np.ndarray) and len(neurons.shape) == 1
		#assert nxt_b_i == len(buf_l)
		assert nxt_w_i == len(weights)
		p_l.append(neurons)
		defineArgMax(smtobj, prefix+'n{}_'.format(nxt_n_i))
		if results is not None:
			#var = 'pred_is_ok_'+str(s_i)
			#defineMaxIsHere(smtobj, var, prefix+'n{}_'.format(nxt_n_i), results[s_i])
			#smtobj.Assert(var)
			var = prefix+'n{}_'.format(nxt_n_i)
			smtobj.Assert([var+'_ArgMax_is_'+str(results[s_i]), 0.0])
	r = np.array(p_l)
	print(r)
	
	smtobj.checkSat()
	smtobj.getValue([prefix+'n{}_{}'.format(nxt_n_i,i) for i in range(len(neurons))] \
				  + ['noise_s','noise_nb'] \
				  + ['noise'+str(i) for i in range(prod(samples.shape[1:]))])
	return smtobj

if __name__ == "__main__":
	sz = 14
	name = '2_layer_mlp_SMALL{}'.format(sz)
	# name = '2_layer_mlp_SMALL{}_1000'.format(sz)
	datafile = 'mnist_small{}.npz'.format(sz)
	#
	fd = open(name+'.json', encoding='utf8')
	nn_js = json.load(fd)
	fd.close()
	if False:
		fd = open(name+'_test.json', encoding='utf8')
		predictions = json.load(fd)
		fd.close()
		N = 20
		preds = np.array(predictions)[:N]
		maxs = preds.argsort() ## pour les 2 plus grands : [:,-1:-3:-1]
		deltas = [[preds[i,maxs[i,-1]] - preds[i,maxs[i,-2]],
				preds[i,maxs[i,-2]] - preds[i,maxs[i,-3]]] for i in range(N)]
		for i,x in enumerate(deltas): print((i,x))
	"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
digits (InputLayer)          [(None, 784)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                50240      (64*785) RELU
_________________________________________________________________
predictions (Dense)          (None, 10)                650        (10*65)
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
	"""
	if True:
		npz = np.load(datafile)
		x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
		npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	i = 3 # 0
	smtobj = translate(nn_js, x_test[i:i+1],y_test[i:i+1])
	#smtobj.to_file(nn_js[0]['name'] + ('_INT' if levels else ''))
	smtobj.to_file(name)
