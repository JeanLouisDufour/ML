import tensorflow as tf
import numpy as np

### scipy.special.softmax : np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def tf_softmax(n):
	""
	tmp = tf.exp(n) ## egal a np.exp(np.float32(x))
	return tf.divide(tmp , tf.reduce_sum(tmp))

