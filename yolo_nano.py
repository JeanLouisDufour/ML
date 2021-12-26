# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

import numpy as np, struct, time
import cv2 as cv
from scipy.ndimage import correlate
import my_cv_dnn

# https://github.com/pjreddie/darknet/blob/master/data/dog.jpg
image, min_confidence, NMS_threshold = "dog.jpg", 0.5, 0.3
image, min_confidence, NMS_threshold = "scream.jpg", 0.5, 0.5
image, min_confidence, NMS_threshold = "horses.jpg", 0.5, 0.3

######################################
blob_shape = (416, 416)
#blob_shape = (208, 416) # -> erreur concat_layer.cpp:95
data_dir = r"E:\github_data\ML" + "\\"
data_dir = r"C:\Users\F074018\Documents\github_data\ML" + "\\"
# https://github.com/pjreddie/darknet/blob/master/data/coco.names
y_n = "coco.names"
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
y_cfg = "yolov3.cfg"
# https://pjreddie.com/media/files/yolov3.weights
y_weights = data_dir+"yolov3.weights"

last_line = 63 # 42 # 33
with open(y_cfg) as fd:
	y_cfg_lines = fd.readlines()
#y_cfg_lines = y_cfg_lines[:last_line]
y_cfg_txt = ''.join(y_cfg_lines).encode('cp1250')

my_net = my_cv_dnn.ReadDarknetFromCfg(y_cfg, CV_450 = False)
my_layers = my_net['layers']

def read_raw_weights(weight_file, bsz):
	""
	with open(weight_file, 'rb') as w_f:
		#fsz = os.fstat(w_f.fileno()).st_size
		major,    = struct.unpack('i', w_f.read(4))
		minor,    = struct.unpack('i', w_f.read(4))
		#revision, = struct.unpack('i', w_f.read(4))

		if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
			#w_f.read(8)
			offset = 20
		else:
			#w_f.read(4)
			offset = 16
		w_f.seek(0,0) # reset
		b = w_f.read(offset+bsz)
	return b

def create_raw_weights(al):
	""
	b = b'\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00|\xe8\x01\x00\x00\x00\x00'
	b = bytearray(b) # mutable
	for a in al:
		for f in a.flat:
			b += struct.pack('f',f)
			assert struct.unpack('f',b[-4:]) == (f,)
	assert len(b) == 20 + sum(np.prod(a.shape) for a in al)*4
	return bytes(b)

bsl =  [(32,3,3,3)]  + [(1,32)]*4
bsl += [(64,32,3,3)] + [(1,64)]*4
bsl += [(32,64,1,1)] + [(1,32)]*4
bsl += [(64,32,3,3)] + [(1,64)]*4
bsz = sum(np.prod(bs) for bs in bsl) * 4

b_conv_0 = [np.ones((32,3,3,3), 'float32')]
b_conv_0 = [np.float32(np.random.standard_normal((32,3,3,3)))]
b_bn_0 = [np.zeros((1,32), 'float32')] * 4
w_buf = create_raw_weights(b_conv_0 + b_bn_0)
w_buf = create_raw_weights([np.array(range(bsz//4), 'float32')])
w_buf = read_raw_weights(y_weights, bsz)
	
LABELS = open(y_n).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print('BEGIN cv.dnn.readNetFromDarknet')
#net = cv.dnn.readNetFromDarknet(y_cfg_txt, w_buf) # , y_weights)
net = cv.dnn.readNetFromDarknet(y_cfg, y_weights)
print('END cv.dnn.readNetFromDarknet')
lnames = ['_input'] + net.getLayerNames()
L = [net.getLayer(i) for i in range(len(lnames))]
assert all(Li.name == li for Li,li in zip(L, lnames))

if False:
	net1 = my_cv_dnn.ReadDarknetFromCfg(y_cfg)
	layers = my_cv_dnn.analyze(net, (1, 3) + blob_shape)
	l1 = [d["layer_name"] for d in net1["layers"]]
	l2 = [l[0] for l in layers]
	assert l1 == l2[1:]
	t1 = [d["layer_type"] for d in net1["layers"]]
	t2 = [l[1] for l in layers]
	assert t1 == t2[1:]
	_ = 2+2

image = cv.imread(image)
print(image.shape, image.dtype)
(H, W) = image.shape[:2]

# WARNING : en opencv, Size(W,H) et resize(W,H)
blob_WH = (blob_shape[1], blob_shape[0])
blob = cv.dnn.blobFromImage(image, 1 / 255.0, blob_WH, swapRB=True, crop=False)
assert blob.shape == (1,3) + blob_shape
blob1 = my_cv_dnn.blobFromImage(image, 1 / 255.0, blob_WH, swapRB=True, crop=False)
assert all((blob == blob1).flatten())
#### test
####
blob = np.ones((1, 3, 416, 416), 'float32')
blob = np.float32(np.random.uniform(size=(1, 3, 416, 416)))
net.setInput(blob)
# determine only the *output* layer names that we need from YOLO
#ln = net.getUnconnectedOutLayersNames()
ln =  lnames[1:] # ['conv_0','bn_0','relu_0']

print(f"[INFO] YOLO starts")
start = time.perf_counter()
layerOutputs = net.forward(ln)
end = time.perf_counter()
print(f"[INFO] YOLO took {end-start} seconds")

def bn_(inp, blobs):
	""
	assert 1 == inp.shape[0]
	[meanData], [stdData], [weightsData], [biasData] = blobs
	assert len(meanData)==len(stdData)==len(weightsData)==len(biasData)==inp.shape[1]
	r_shape = inp.shape
	r = np.empty(r_shape, 'float32')
	for p in range(inp.shape[1]):
		w = weightsData[p] / np.sqrt(stdData[p]+1e-6)
		b = biasData[p] - w * meanData[p]
		r[0,p,:,:] = inp[0,p,:,:] * w + b
	return r

filter2D_manu = False
filter2D_scipy = True
def filter2D(img_2d, ker_2d, out = None):
	""
	assert filter2D_manu or filter2D_scipy
	img_out = out
	H,W = img_2d.shape
	if img_out is None:
		img_out = np.empty(img_2d.shape, img_2d.dtype)
	else:
		assert img_out.shape == img_2d.shape and img_out.dtype == img_2d.dtype
	assert ker_2d.shape == (3,3)
	if not filter2D_manu:
		H = 0
	for li in range(H):
		LI = -1 if li==0 else +1 if li==H-1 else 0
		for co in range(W):
			CO = -1 if co==0 else +1 if co==W-1 else 0
			x = 0
			if True: #for plane in range(C):
				#img_2d = img_3d[plane]
				#ker_2d = ker_3d[plane]
				for i in range(3):
					true_li = li + (i-1)
					true_li_is_ok = 0 <= true_li < H
					for j in range(3):
						true_co = co + (j-1)
						true_co_is_ok = 0 <= true_co < W
						pixel = img_2d[true_li, true_co] if true_li_is_ok and true_co_is_ok else 0
						x += pixel * ker_2d[i,j]
			img_out[li,co] = x
	if filter2D_scipy:
		img_out_1 = correlate(img_2d, ker_2d, output=None, mode='constant', cval=0.0)
	if filter2D_manu and filter2D_scipy:
		print('filter2D error : ', np.max(np.abs(img_out_1 - img_out).flat))
	if not filter2D_manu:
		img_out[:,:] = img_out_1[:,:]
	return img_out

def conv_(inp, blobs, out = None, params={}):
	""
	stride = params.get('stride',1)
	assert stride in (1,2)
	assert 1 == inp.shape[0] and inp.shape[2] == inp.shape[3]
	if len(blobs) == 1:
		[kernels] = blobs
		bias = None
	else:
		[kernels, bias] = blobs # conv_81: (1,255=po)
		assert bias.shape == (1, kernels.shape[0])
	assert (bias is not None) == params['bias_term']
	po, pi, kx, ky = kernels.shape
	assert pi == inp.shape[1] and kx==ky and kx in (1,3)
	tmp = np.empty(inp.shape[1:], 'float32')
	tmp_sum = np.empty(inp.shape[2:], 'float32')
	assert inp.shape[2] % stride == 0 and inp.shape[3] % stride == 0
	r_shape = (1, po, inp.shape[2]//stride, inp.shape[3]//stride)
	r = out
	if r is None:
		r = np.empty(r_shape, 'float32')
	else:
		assert r.shape == r_shape
	for p in range(po):
		cur_r = r[0,p]
		for q in range(pi):
			cur_tmp = tmp[q]
			cur_inp = inp[0,q]
			cur_ker = kernels[p,q]
			if kx == 1:
				np.multiply(cur_inp, cur_ker[0,0], out=cur_tmp)
			else:
				filter2D(cur_inp, cur_ker, out=cur_tmp)
		tmp_sum = np.sum(tmp, axis=0)
		if bias is not None:
			tmp_sum += bias[0,p]
		if stride == 1:
			r[0,p,:,:] = tmp_sum
		else:
			r[0,p,:,:] = tmp_sum[::2,::2]
	return r

relu_coeff = np.float32(0.1)
def relu_(inp, blobs=[], out=None):
	""
	assert blobs == []
	if out is None:
		out = np.empty(inp.shape, inp.dtype)
	else:
		assert out.shape == inp.shape and out.dtype == inp.dtype
	#return np.vectorize(lambda x : x if x >= 0 else relu_coeff*x)(inp)
	np.multiply(inp, relu_coeff, out=out)
	np.maximum(inp, out, out=out) # alias supporte
	return out # np.maximum(inp, relu_coeff*inp)

one_f32 = np.float32(1)
half_f32 = np.float32(0.5)
logistic_activate = lambda x: one_f32 / (one_f32 + np.exp(-x))

prev_name = None
out_d = {}
inp = blob # init : shape = (1,3,h,w)
for idx, (lo,Li,mLi) in enumerate(zip(layerOutputs, L[1:], my_layers[:len(layerOutputs)]), start=1):
	# lo.shape == (1,c,h,w), sauf sur un Region : (N,85)
	print(Li.name)
	assert Li.type == mLi['layer_type']
	assert Li.name == mLi['layer_name'], (Li.name , mLi['layer_name'])
	params = mLi['layerParams']
	#if not Li.name.startswith('yolo_'):
	#if not params.get('bias_term', False):
	#if Li.name.startswith('conv_'):
	if False:
		lo_estim = lo
	elif Li.type == 'BatchNorm':
		assert Li.name.startswith('bn_')
		assert inp.shape == lo.shape
		lo_estim = bn_(inp, Li.blobs)
	elif Li.type == 'Concat':
		assert Li.name.startswith('concat_')
		assert params['axis'] == 1
		n1,n2 = mLi['bottom_indexes']
		assert n1 == prev_name
		inp2 = out_d[n2]
		assert inp.shape[0] == inp2.shape[0] == 1 and inp.shape[2:] == inp2.shape[2:]
		lo_estim = np.empty((1, inp.shape[1] + inp2.shape[1]) + inp.shape[2:], 'float32')
		lo_estim[0,:inp.shape[1],:,:] = inp
		lo_estim[0,inp.shape[1]:,:,:] = inp2
	elif Li.type == 'Convolution':
		assert Li.name.startswith('conv_')
		assert 1 == inp.shape[0] == lo.shape[0]
		assert inp.shape[2] == inp.shape[3] and lo.shape[2] == lo.shape[3]
		if inp.shape[2] == lo.shape[2]:
			stride = 1
		else:
			assert inp.shape[2] == 2*lo.shape[2]
			stride = 2
		assert stride == params['stride']
		lo_estim = conv_(inp, Li.blobs, params=params)
	elif Li.type == 'Eltwise':
		assert Li.name.startswith('shortcut_')
		assert params['op'] == 'sum'
		n1,n2 = mLi['bottom_indexes']
		assert n2 == prev_name
		inp1 = out_d[n1]
		assert inp1.shape == inp.shape
		lo_estim = inp+inp1
	elif Li.type == 'Identity':
		assert Li.name.startswith('identity_')
		[n1] = mLi['bottom_indexes']
		lo_estim = out_d[n1]
	elif Li.type == 'Permute':
		assert Li.name.startswith('permute_') # permute_82 : order = [0,2,3,1]
		order = params['order']
		assert order == [0,2,3,1]
		lo_estim = np.transpose(inp, order)
	elif Li.type == 'Region':
		assert Li.name.startswith('yolo_')
		n1,n2 = mLi['bottom_indexes']
		assert n1 == prev_name and n2 == 'data'
		inp_data = blob # 1,3,416,416
		[b] = params['blobs']
		[[b_f32]] = Li.blobs
		assert len(b) == 6 and (b_f32 == b).all() # 6 == 2*3
		################# constructeur RegionLayerImpl ###########
		thresh = params.get("thresh", 0.2)
		coords = params.get("coords", 4)
		anchors = params['anchors']           # 3
		classes = params['classes']			# 80
		classfix = params.get("classfix", 0)
		useSoftmax = params.get("softmax", False)
		useLogistic = params.get("logistic", False)  # True
		nmsThreshold = params.get("nms_threshold", 0.4)
		scale_x_y = params.get("scale_x_y", 1.0) # Yolov4
		assert nmsThreshold >= 0 and coords ==4 and (useLogistic and not useSoftmax) and anchors == 3 and classes == 80
		# yolo_82 : inp : (1,13,13,255) -> (507, 85)
		# yolo_94 : inp : (1,26,26,255) -> (2028, 85)
		# yolo_106 : ...
		################### .getMemoryShapes ################
		batch_size = inp.shape[0] # 1
		assert batch_size == 1
		assert inp.shape[3] == (1+coords+classes) * anchors
		lo_estim = np.zeros((np.prod(inp.shape[1:3]) * anchors, inp.shape[3] // anchors), 'float32')
		################## .forward ################
		# cell : 1 + 4 + 80, et il y en a rows*cols*anchors
		cell_size = classes + coords + 1 # 85
		biasData = b_f32
		rows, cols = inp.shape[1:3]
		hNorm, wNorm = inp_data.shape[2:]
		sample_size = cell_size*rows*cols*anchors
		assert sample_size == inp.size
		assert sample_size == lo_estim.size
		inp_r = inp.reshape((rows*cols*anchors, cell_size))
		for x in range(cols):
			for y in range(rows):
				for a in range(anchors):
					i = (y*cols + x)*anchors + a
					lo_estim[i,4] = logistic_activate(inp_r[i,4])
					lo_estim[i,5:] = logistic_activate(inp_r[i,5:])
					scale = lo_estim[i,4]
					x_tmp = (logistic_activate(inp_r[i,0]) - half_f32) * scale_x_y + half_f32
					y_tmp = (logistic_activate(inp_r[i,1]) - half_f32) * scale_x_y + half_f32
					lo_estim[i,0] = (x + x_tmp) / cols
					lo_estim[i,1] = (y + y_tmp) / rows
					lo_estim[i,2] = np.exp(inp_r[i,2]) * biasData[2 * a] / wNorm
					lo_estim[i,3] = np.exp(inp_r[i,3]) * biasData[2 * a + 1] / hNorm
					for j in range(classes):
						prob = scale*lo_estim[i,j+5]
						lo_estim[i,j+5] = prob if (prob > thresh) else 0
		assert nmsThreshold == 0
	elif Li.type == 'ReLU':
		assert Li.name.startswith('relu_')
		assert inp.shape == lo.shape
		lo_estim = relu_(inp)
	elif Li.type == 'Resize':
		assert Li.name.startswith('upsample_')
		assert params['zoom_factor'] == 2
		assert inp.shape[0] == 1
		lo_estim = np.empty((1, inp.shape[1], inp.shape[2]*2, inp.shape[3]*2), 'float32')
		lo_estim[0,:,::2,::2] = inp
		lo_estim[0,:,::2,1::2] = inp
		lo_estim[0,:,1::2,::2] = inp
		lo_estim[0,:,1::2,1::2] = inp
	else:
		assert False, (Li.type, Li.name)
	err = np.max(np.abs(lo_estim - lo).flat)
	print(err)
	if err > 3e-5:
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	inp = out_d[Li.name] = lo
	prev_name = Li.name

l0,l1,l2 = layerOutputs

conv_0_out = img_out_0 = l0[0][0] # (416,416)

##################
# correlate
#######

img_3d = blob[0] # shape = (3,h,w)
ker_3d = L[1].blobs[0][0] # shape = (3,3,3)

def filter3D(img_3d, ker_3d):
	""
	C,H,W = img_3d.shape
	img_out = np.empty((H,W), 'float32')
	assert ker_3d.shape == (C,3,3)
	for li in range(H):
		LI = -1 if li==0 else +1 if li==H-1 else 0
		for co in range(W):
			CO = -1 if co==0 else +1 if co==W-1 else 0
			x = 0
			for plane in range(C):
				img_2d = img_3d[plane]
				ker_2d = ker_3d[plane]
				for i in range(3):
					true_li = li + (i-1)
					true_li_is_ok = 0 <= true_li < H
					for j in range(3):
						true_co = co + (j-1)
						true_co_is_ok = 0 <= true_co < W
						pixel = img_2d[true_li, true_co] if true_li_is_ok and true_co_is_ok else 0
						x += pixel * ker_2d[i,j]
			img_out[li,co] = x
	return img_out

if False:
	img_out_1 = filter3D(img_3d, ker_3d)
	print(np.max(np.abs(img_out_1 - conv_0_out).flat))

if True:
	# scipy.ndimage.correlate renvoie un (3,H,W)
	img_out_2 = correlate(img_3d, ker_3d, output=None, mode='constant', cval=0.0)[1]
	print(np.max(np.abs(img_out_2 - conv_0_out).flat))

bn_0_out = l1[0][0]
[meanData], [stdData], [weightsData], [biasData] = L[2].blobs
w0 = weightsData[0] / np.sqrt(stdData[0]+1e-6)
b0 = biasData[0] - w0 * meanData[0]
# bn_0_out_estim = conv_0_out * 9.177355 + 0.9320135
bn_0_out_estim = conv_0_out * w0 + b0
print(np.max(np.abs(bn_0_out_estim - bn_0_out).flat))

relu_0_out = l2[0][0]
relu_0_out_estim = np.vectorize(lambda x : x if x >= 0 else np.float32(0.1)*x)(bn_0_out)
print(np.max(np.abs(relu_0_out_estim - relu_0_out).flat))