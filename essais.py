import json, numpy as np
import matplotlib.pyplot as plt
# pour plt.imshow(x_train[1])
#import skimage
# pour skimage.transform.resize et skimage.transform.rescale

def resize(src, dst=None, binary=False, sz=14):
	""
	assert 10 <= sz <= 14
	i0 = j0 = 14-sz
	a = 2
	#sz = 14; i0=j0=0
	#sz = 12; i0=j0=2
	assert i0==j0 and a*sz+i0+j0 == 28
	assert src.shape == (28,28) and src.dtype == np.uint8
	if dst is None:
		dst = np.zeros((sz,sz), dtype=np.uint8)
	else:
		assert dst.shape == (sz,sz) and dst.dtype == np.uint8
	for i in range(sz):
		for j in range(sz):
			s = sum(src[i0+a*i+k,j0+a*j+l] for k in range(a) for l in range(a))
			if binary:
				dst[i,j] = 0 if s < (a*255 / 2) else 255
			else:
				dst[i,j] = s // (a*a)
	return dst

if __name__ == "__main__":
	name = '2_layer_mlp'
	try: nn_js
	except NameError:
		fd = open(name+'.json', encoding='utf8')
		nn_js = json.load(fd)
		fd.close()
	try: predictions
	except NameError:
		fd = open(name+'_test.json', encoding='utf8')
		predictions = json.load(fd)
		fd.close()
	
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
	try: x_train0
	except NameError:
		npz = np.load('mnist.npz')
		x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
		npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	##
	N = 40
	preds = np.array(predictions)[:N]
	maxs = preds.argsort() ## pour les 2 plus grands : [:,-1:-3:-1]
	deltas = [[preds[i,maxs[i,-1]] - preds[i,maxs[i,-2]],
			preds[i,maxs[i,-2]] - preds[i,maxs[i,-3]]] for i in range(N)]
	##
	image = x_train0[5]
	assert image.shape == (28,28) and image.dtype == np.uint8
	#image1 = np.zeros((14,14), dtype=np.uint8)
	image1 = resize(image)
	plt.imshow(image1, cmap='gray')
	
	sz = 12
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
