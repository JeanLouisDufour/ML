import tensorflow as tf
import numpy as np

### scipy.special.softmax : np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def tf_softmax(n):
	""
	tmp = tf.exp(n) ## egal a np.exp(np.float32(x))
	return tf.divide(tmp , tf.reduce_sum(tmp))

"""
neurons1a = np.exp(neurons1a - np.max(neurons1a))
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
"""