"""
genere les versions "quantifiees" des NN traitant mnist
"""
import json
import numpy as np
import matplotlib.pyplot as plt
# pour plt.imshow(x_train[1], cmap='gray')
from nnjs import predict, quant

if __name__ == "__main__":
	sz = 14
	name = '2_layer_mlp_SMALL{}'.format(sz)
	datafile = 'mnist_small{}.npz'.format(sz)
	#
	fd = open(name+'.json', encoding='utf8')
	nn_js = json.load(fd)
	fd.close()
	if True:
		npz = np.load(datafile)
		x_train0, y_train, x_test0, y_test = (npz[s] for s in ('x_train', 'y_train', 'x_test', 'y_test'))
		npz.close()
		x_train, x_test = x_train0.astype('float32') / 255, x_test0.astype('float32') / 255
	##
	pr = predict(nn_js, x_test.reshape(10000, sz*sz))
	estim_ref = pr.argmax(axis=1)
	print('bad estims : ' + str(sum(estim_ref != y_test)))
	for bits in range(4,9):
		print('bits : '+str(bits))
		levels = 1 << bits
		max_denom = [1]
		nn_js_quant = quant(nn_js, levels, max_denom)
		pr = predict(nn_js_quant, x_test.reshape(10000, sz*sz))
		estim = pr.argmax(axis=1)
		print(sum(estim != estim_ref))
		print(max_denom[0])
