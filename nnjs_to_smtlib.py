import json, numpy as np
from smtlib import smtlib
#
import functools, operator
prod = lambda l: functools.reduce(operator.mul,l,1)

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
		smtobj.defineFun(name+str(i),[],'Real',float(ai))
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
		se = [['*', prefix_in+str(j), float(Aji)] for j,Aji in enumerate(A[:,i])]
		se = ['+', float(Bi)] + se
		smtobj.defineFun(prefix_out+str(i),[],'Real',se)
	defineArray_d[prefix_out] = len(B)

def defineArrayReLU(smtobj, prefix_out, prefix_in, sz):
	""
	sz_in = defineArray_d[prefix_in]
	assert sz_in == sz
	for i in range(sz):
		inp = prefix_in+str(i)
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

def translate(nn_js, samples):
	"""
	
	"""
	smtobj = smtlib('QF_LIRA')
	smtobj.setOption(':produce-models','true')
	configuration, weights = nn_js
	# compil
	#buf_l = [np.zeros(sh,dtype=np.float32) for sh in neuron_shapes[1:]]
	#
	p_l = []
	for s_i, sample in enumerate(samples):
		prefix = 's'+str(s_i)
		neurons = sample
		defineArray(smtobj, prefix+'n0_', neurons)
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
	r = np.array(p_l)
	print(r)
	smtobj.checkSat()
	smtobj.getValue([prefix+'n{}_{}'.format(nxt_n_i,i) for i in range(len(neurons))])
	return smtobj

if __name__ == "__main__":
	fd = open('2_layer_mlp.json', encoding='utf8')
	nn_js = json.load(fd)
	fd.close()
	try: x_train0
	except NameError:
		npz = np.load('mnist.npz')
		x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
		npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	smtobj = translate(nn_js, x_train[:1])
	smtobj.to_file(nn_js[0]['name'])
