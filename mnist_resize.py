"""
genere les versions "small" de mnist
"""

import numpy as np
import matplotlib.pyplot as plt
# pour plt.imshow(x_train[1], cmap='gray')
#import skimage
# pour skimage.transform.resize et skimage.transform.rescale

def resize(src, dst=None, minmax=True, sz=14):
	""
	assert 1 <= sz <= 28
	a,r = divmod(28,sz)
	i0 = j0 = r//2
	#sz = 14; i0=j0=0
	#sz = 12; i0=j0=2
	assert i0==j0 and a*sz+2*i0 in (27,28)
	assert src.shape == (28,28) and src.dtype == np.uint8
	if dst is None:
		dst = np.zeros((sz,sz), dtype=np.uint8)
	else:
		assert dst.shape == (sz,sz) and dst.dtype == np.uint8
	buf = np.zeros(a*a, dtype=np.uint8)
	for i in range(sz):
		for j in range(sz):
			#buf[:] = np.fromiter((src[i0+a*i+k,j0+a*j+l] for k in range(a) for l in range(a)), dtype=np.uint8)
			buf[:] = src[i0+a*i:i0+a*i+a,j0+a*j:j0+a*j+a].reshape(a*a)
			if minmax:
				dst[i,j] = np.min(buf) if np.mean(buf) < 128 else np.max(buf)
			else:
				dst[i,j] = np.mean(buf)
	return dst

if __name__ == "__main__":

	if True:
		npz = np.load('mnist.npz')
		x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
		npz.close()
	##
	for sz in range(4,7):
		print(sz)
		print('x_train0')
		N = x_train0.shape[0]
		x_train0_small = np.zeros((N,sz,sz), dtype=np.uint8)
		for i in range(N):
			resize(x_train0[i], x_train0_small[i], sz=sz)
		print('x_test0')
		N = x_test0.shape[0]
		x_test0_small = np.zeros((N,sz,sz), dtype=np.uint8)
		for i in range(N):
			resize(x_test0[i], x_test0_small[i], sz=sz)
		print('saving')
		np.savez('mnist_small{}.npz'.format(sz), x_train=x_train0_small, y_train=y_train, x_test=x_test0_small, y_test=y_test)
#		for i in range(10):
#			print(i)
#			plt.imshow(x_test0[i], cmap='gray')
#			plt.imshow(x_test0_small[i], cmap='gray')
