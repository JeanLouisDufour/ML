import json, os
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import matplotlib.pyplot as plt
# pour plt.imshow(x_train[1])
import functools, operator
from tf_experiments import tf_softmax
from nnjs import predict, approx
prod = lambda l: functools.reduce(operator.mul,l,1)
assert prod(range(1,4)) == 6

NoneType = type(None)

def clean(o):
	""
	if isinstance(o, dict):
		for k,v in o.items():
			o[k] = clean(v)
	elif isinstance(o,list):
		for k,v in enumerate(o):
			o[k] = clean(v)
	elif isinstance(o, (bool,int,float,NoneType,str)):
		pass
	elif isinstance(o, tuple):
		o = [clean(v) for v in o]
	else:
		assert False, o
	return o


def activation_relu(neurons):
	""
	pass

def INH_predict(nn_js, samples):
	""
	configuration, weights = nn_js
	# compil
	#buf_l = [np.zeros(sh,dtype=np.float32) for sh in neuron_shapes[1:]]
	#
	p_l = []
	for sample in samples:
		neurons = sample
		nxt_w_i = 0
		for l_i, conf_layer in enumerate(configuration['layers']):
			assert isinstance(neurons, np.ndarray) # and len(neurons.shape) == 1
			kind = conf_layer['class_name']
			if kind == 'Dense':
				w1_,w2_ = weights[nxt_w_i:nxt_w_i+2]
				w1 = np.array(w1_, np.float32); w2 = np.array(w2_, np.float32); 
				nxt_w_i += 2
				sh1 = w1.shape
				assert neurons.shape == sh1[:1]
				# 'a' = np, 'b'= tf
				neurons1a = np.matmul(neurons,w1)
				neurons1b = tf.matmul(neurons.reshape((1,len(neurons))),w1)
				assert all(neurons1a == neurons1b[0].numpy())
				neurons1a += w2
				neurons1b = tf.nn.bias_add(neurons1b, w2)
				assert all(neurons1a == neurons1b[0].numpy())
				activ = conf_layer['config']['activation']
				if activ == 'relu':
					neurons1a = np.fromiter(((0 if x < 0 else x) for x in neurons1a), np.float32)
					neurons1b = tf.nn.relu(neurons1b)
					assert all(neurons1a == neurons1b[0].numpy())
				elif activ == 'softmax':
					neurons1a = np.exp(neurons1a)
					# neurons1a *= (np.float32(1)/ np.sum(neurons1a))
					# neurons1a *= (np.float64(1)/ np.sum(neurons1a.astype(np.float64)))
					neurons1a /= np.sum(neurons1a)
					if abs(np.sum(neurons1a)-1) >= 1.2e-7:
						print('???? 1a norm err : {}'.format(abs(np.sum(neurons1a)-1)))
					tmp2 = tf_softmax(neurons1b)
					neurons1b = tf.nn.softmax(neurons1b)
					if abs(tmp2 - neurons1b).max() >= 1e-7:
						 print('???? softmax built-in vs manu_tf : {}'.format(abs(tmp2 - neurons1b).max()))
					if tf.reduce_sum(neurons1b).numpy() != np.sum(neurons1b):
						print('???? tf vs. np : {} vs {}'.format(tf.reduce_sum(neurons1b).numpy() - 1,np.sum(neurons1b) - 1))
					if abs(tf.reduce_sum(neurons1b).numpy() - 1) >= 1.2e-7:
						print('???? 1b norm tf err : {}'.format(tf.reduce_sum(neurons1b).numpy() - 1))
					if abs(np.sum(neurons1b) - 1) >= 1.2e-7:
						print('???? 1b norm np err : {}'.format(np.sum(neurons1b) - 1))
					errs = np.abs(neurons1a - neurons1b[0].numpy()) / (neurons1a + neurons1b[0].numpy())
					if np.max(errs) >= 1e-6:
						print('???? 1a - 1b max rel err : {}'.format(np.max(errs)))
				elif activ == 'linear':
					_ = 2+2
				else:
					assert False, activ
				neurons = neurons1a
			elif kind == 'Dropout':
				_ = 2+2
			elif kind == 'Flatten':
				d1,d2 = neurons.shape
				neurons = neurons.reshape((d1*d2,))
			elif kind == 'InputLayer':
				_ = 2+2
			else:
				assert False, kind
			assert isinstance(neurons, np.ndarray) and len(neurons.shape) == 1
		#assert nxt_b_i == len(buf_l)
		assert nxt_w_i == len(weights)
		p_l.append(neurons)
	r = np.array(p_l)
	return r

def chk(model, write = False):
	"""
	"""
	assert isinstance(model, tf.keras.Model)
	# config globale
	configuration = model.get_config()
	#clean(configuration)
	#assert configuration == json.loads(json.dumps(configuration))
	conf_layer_l = configuration['layers']
	#
	layer_l = model.layers
	assert len(layer_l) == len(conf_layer_l)
	neuron_shapes = []
	nb_weights = 0
	for l_i, (layer, conf_layer) in enumerate(zip(layer_l, conf_layer_l)):
		config = layer.get_config()
		assert config == conf_layer['config']
		assert config['dtype'] == 'float32'
		weights = layer.get_weights()
		assert isinstance(weights, list)
		if isinstance(layer, tf.keras.layers.Dense):
			""" Just your regular densely-connected NN layer.
Dense implements the operation: output = activation(dot(input, kernel) + bias)
where activation is the element-wise activation function passed as the activation argument,
kernel is a weights matrix created by the layer,
and bias is a bias vector created by the layer (only applicable if use_bias is True).
Note: if the input to the layer has a rank greater than 2,
then it is flattened prior to the initial dot product with kernel.
			"""
			assert l_i > 0
			assert conf_layer['class_name'] == 'Dense'
			assert config['use_bias']
			assert config['bias_constraint']==config['bias_regularizer']==None
			assert config['bias_initializer']['class_name'] == 'Zeros'
			assert config['kernel_constraint']==config['kernel_regularizer']==None
			assert config['kernel_initializer']['class_name'] == 'GlorotUniform'
			assert config['activation'] in ('linear','relu','softmax'), config
			sz = config['units'] # 10
			assert len(weights) == 2
			nb_weights += 2
			sh0 = weights[0].shape
			sh1 = weights[1].shape
			assert sh0[1:] == sh1 == (sz,)  # (128,10) et (10,)
			assert sh0[0] == neuron_shapes[-1][0]
			neuron_shapes.append([sz,])
		elif isinstance(layer, tf.keras.layers.Dropout):
			""" Applies Dropout to the input.
Dropout consists in randomly setting a fraction rate of input units to 0
at each update during training time, which helps prevent overfitting.
			"""
			assert l_i > 0
			assert conf_layer['class_name'] == 'Dropout'
			assert weights == []
			assert config['noise_shape'] is None
			if config['seed'] is None:
				print('WARNING : Dropout without seed')
			assert 0 < config['rate'] <= 1
			### no predict action
		elif isinstance(layer, tf.keras.layers.Flatten):
			assert l_i == 0
			assert conf_layer['class_name'] == 'Flatten'
			assert weights == []
			assert config['data_format'] == 'channels_last'
			batch_input_shape = config['batch_input_shape'] # [None, 28,28]
			assert batch_input_shape[0] is None
			neuron_shapes.append(list(batch_input_shape[1:]))
			neuron_shapes.append([prod(neuron_shapes[-1])])
		elif isinstance(layer, tf.keras.layers.InputLayer):
			assert l_i == 0
			assert conf_layer['class_name'] == 'InputLayer'
			assert weights == []
			assert config['ragged'] == config['sparse'] == False
			batch_input_shape = config['batch_input_shape'] # [None, 784]
			assert batch_input_shape[0] is None
			assert len(batch_input_shape) == 2
			neuron_shapes.append(list(batch_input_shape[1:]))
		else:
			assert False, layer
		_ = 2+2
	# weights
	weights_orig = model.get_weights()
	assert isinstance(weights_orig, list)
	assert len(weights_orig) == nb_weights
	weights = weights_orig.copy()
	for w_i, w in enumerate(weights):
		assert isinstance(w,np.ndarray)
		assert w.dtype == np.float32
		wsh = list(w.shape)
		assert len(wsh) in (1,2)
		wpy = w.tolist()
		w1 = np.array(wpy)
		if len(wsh) == 1:
			assert all(w == w1)
		else:
			assert all(all(bl) for bl in (w == w1))
		assert wpy == json.loads(json.dumps(wpy))
		weights[w_i] = wpy
	#
	clean(configuration)
	assert configuration == json.loads(json.dumps(configuration))
	nn_js = [configuration, weights]
	if write:
		fd = open(configuration['name']+'.json', 'w')
		json.dump(nn_js,fd)
		fd.close()
	return nn_js

dbg = True
chk_determ = False
N = 10000

def mnist_small(sz):
	# MNIST SMALL
	#sz = 12
	if True:
		if True:
			npz = np.load('mnist_small{}.npz'.format(sz))
			x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
			npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	#################################
	if True:
		print('********** BEG model0 **********')
		inputs = tf.keras.Input(shape=(sz*sz,), name='digits')
		x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
		#x = layers.Dense(64, activation='relu', name='dense_2')(x)
		outputs = tf.keras.layers.Dense(10, name='predictions')(x)
		model = tf.keras.Model(inputs=inputs, outputs=outputs, name='2_layer_mlp_SMALL{}'.format(sz))
		if dbg:
			model.summary()
		"""
Model: "2_layer_mlp"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
digits (InputLayer)          [(None, 196)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                12608     
_________________________________________________________________
predictions (Dense)          (None, 10)                650       
=================================================================
Total params: 13,258
Trainable params: 13,258
Non-trainable params: 0
_________________________________________________________________
		"""
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop())
		history = model.fit(x_train.reshape(60000, sz*sz), y_train,
                    batch_size=64,
                    epochs=10)
		#print('\nhistory dict:', history.history)
		"""
Train on 60000 samples
60000/60000 [==============================] - 4s 70us/sample - loss: 0.3486
		"""
		model0 = model
		print('********** END model0 **********')
	# conversion
	nn_js = chk(model0)
	nn_js_ap = approx(nn_js, 16384)
	# verif sur train
	print('verif sur train')
	pred0 = model0.predict(x_train.reshape(60000, sz*sz))
	estim0 = pred0.argmax(axis=1)
	print('success : {}'.format(sum(estim0 == y_train)))
	pred0_1 = predict(nn_js, x_train.reshape(60000, sz*sz))
	estim0_1 = pred0_1.argmax(axis=1)
	assert all(estim0 == estim0_1)
	err = abs((pred0 - pred0_1) / (pred0 + pred0_1)).max().max()
	print('err max : {}'.format(err))
	pred0_2 = predict(nn_js_ap, x_train.reshape(60000, sz*sz))
	estim0_2 = pred0_2.argmax(axis=1)
	print('err nb approx : {}'.format(sum(estim0 != estim0_2)))
	err = abs((pred0 - pred0_2) / (pred0 + pred0_2)).max().max()
	print('err max approx : {}'.format(err))
	# test
	print('test')
	pred0 = model0.predict(x_test.reshape(10000, sz*sz))
	estim0 = pred0.argmax(axis=1)
	print('success : {}'.format(sum(estim0 == y_test)))
	pred0_1 = predict(nn_js, x_test.reshape(10000, sz*sz))
	estim0_1 = pred0_1.argmax(axis=1)
	assert all(estim0 == estim0_1)
	err = abs((pred0 - pred0_1) / (pred0 + pred0_1)).max().max()
	print('err max : {}'.format(err))
	pred0_2 = predict(nn_js_ap, x_test.reshape(10000, sz*sz))
	estim0_2 = pred0_2.argmax(axis=1)
	print('err nb approx : {}'.format(sum(estim0 != estim0_2)))
	err = abs((pred0 - pred0_2) / (pred0 + pred0_2)).max().max()
	print('err max approx : {}'.format(err))
	#
	if True:
		fn = nn_js[0]['name']+'_test.json'
		print('writing '+fn)
		fd = open(fn, 'w')
		json.dump(pred0_1.tolist(),fd)
		fd.close()


if __name__ == "__main__":
	for sz in range(8,15):
		mnist_small(sz)

if False:
	## MNIST 
	try: x_train0
	except NameError:
		if False:
			(x_train0, y_train), (x_test0, y_test) = tf.keras.datasets.mnist.load_data(path=os.getcwd()+r'\mnist.npz')
		else:
			npz = np.load('mnist.npz')
			x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
			npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	#################################
	## perso
	try: model0
	except NameError:
		print('********** BEG model0 **********')
		inputs = tf.keras.Input(shape=(784,), name='digits')
		x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
		#x = layers.Dense(64, activation='relu', name='dense_2')(x)
		outputs = tf.keras.layers.Dense(10, name='predictions')(x)
		model = tf.keras.Model(inputs=inputs, outputs=outputs, name='2_layer_mlp')
		if dbg:
			model.summary()
		"""
Model: "2_layer_mlp"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
digits (InputLayer)          [(None, 784)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                50240     
_________________________________________________________________
predictions (Dense)          (None, 10)                650       
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
		"""
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop())
		history = model.fit(x_train.reshape(60000, 784), y_train,
                    batch_size=64,
                    epochs=1)
		print('\nhistory dict:', history.history)
		"""
Train on 60000 samples
60000/60000 [==============================] - 4s 70us/sample - loss: 0.3486
		"""
		model0 = model
		print('********** END model0 **********')
	print(y_test[:N])
	pred0 = model0.predict(x_test.reshape(10000, 784)[:N])
	estim = pred0.argmax(axis=1)
	print(sum(estim == y_test[:N]))
	model0.evaluate(x_test.reshape(10000, 784)[:N], y_test[:N])
	if chk_determ:
		pred0_1 = model0.predict(x_test.reshape(10000, 784)[:N])
		assert all(all(bl) for bl in pred0 == pred0_1)
	nn_js = chk(model0)
	pred0_1 = predict(nn_js, x_test.reshape(10000, 784)[:N])
	estim_1 = pred0_1.argmax(axis=1)
	assert all(estim == estim_1)
	print(sum(estim_1 == y_test[:N]))
	print(abs((pred0 - pred0_1) / (pred0 + pred0_1)).max().max())
	if True:
		fd = open(nn_js[0]['name']+'_test.json', 'w')
		json.dump(pred0_1.tolist(),fd)
		fd.close()
	tf.keras.backend.clear_session()
	
	assert False, "THE END"
	#################################
	## https://www.tensorflow.org/guide/keras/save_and_serialize
	try: model1
	except NameError:
		print('********** BEG model1 **********')
		inputs = tf.keras.Input(shape=(784,), name='digits')
		x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
		x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
		outputs = tf.keras.layers.Dense(10, name='predictions')(x)
		model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
		if dbg:
			model.summary()
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop())
		#x_train = x_train0.reshape(60000, 784).astype('float32') / 255
		#x_test = x_test0.reshape(10000, 784).astype('float32') / 255
		history = model.fit(x_train.reshape(60000, 784), y_train,
                    batch_size=64,
                    epochs=1)
		print('\nhistory dict:', history.history)
		# Reset metrics before saving so that loaded model has same state,
		# since metric states are not preserved by Model.save_weights
		#model.reset_metrics()
		model1 = model
		print('********** END model1 **********')
	print(y_test[:N])
	pred1 = model1.predict(x_test.reshape(10000, 784)[:N])
	#print(pred1)
	print(sum(pred1.argmax(axis=1) == y_test[:N]))
	model1.evaluate(x_test.reshape(10000, 784)[:N], y_test[:N])
	if chk_determ:
		pred1_1 = model1.predict(x_test.reshape(10000, 784)[:N])
		assert all(all(bl) for bl in pred1 == pred1_1)
	nn_js = chk(model1)
	pred1_1 = predict(nn_js, x_test.reshape(10000, 784)[:N])
	print(abs((pred1 - pred1_1) / (pred1 + pred1_1)).max().max())
	tf.keras.backend.clear_session()
	####################
	## https://www.tensorflow.org/overview
	try: model2
	except NameError:
		print('********** BEG model2 **********')
		model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10, activation='softmax')
			])
		if dbg:
			model.summary()
		model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
		# x_train, x_test = x_train0 / 255.0, x_test0 / 255.0
		history = model.fit(x_train, y_train, epochs=5)
		print('\nhistory dict:', history.history)
		model2 = model
		print('********** END model2 **********')
	#########
	print(y_test[:N])
	pred2 = model2.predict(x_test[:N])
	#print(pred2)
	print(sum(pred2.argmax(axis=1) == y_test[:N]))
	model2.evaluate(x_test[:N], y_test[:N])
	if chk_determ:
		pred2_1 = model2.predict(x_test[:N])
		assert all(all(bl) for bl in pred2 == pred2_1)
	nn_js = chk(model2)
	pred2_1 = predict(nn_js, x_test[:N])
	print(abs((pred2 - pred2_1) / (pred2 + pred2_1)).max().max())
	tf.keras.backend.clear_session()
