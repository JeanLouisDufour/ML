import numpy as np, os
import struct
from math import prod

def read_weights(weight_file, bsll=None):
	""
	with open(weight_file, 'rb') as w_f:
		fsz = os.fstat(w_f.fileno()).st_size
		major,    = struct.unpack('i', w_f.read(4))
		minor,    = struct.unpack('i', w_f.read(4))
		revision, = struct.unpack('i', w_f.read(4))

		if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
			w_f.read(8)
			offset = 20
		else:
			w_f.read(4)
			offset = 16

		#??? transpose = (major > 1000) or (minor > 1000)
		if bsll is None:
			#binary = w_f.read()
			all_weights = np.frombuffer(w_f.read(), dtype='float32')
		else:
			bsz = blob_sz(bsll) * 4
			if bsz+offset != fsz:
				assert False, ("bad sizes", fsz, offset, bsz)
			all_weights = [[np.frombuffer(w_f.read(prod(bs)*4), dtype='float32').reshape(bs) for bs in bsl] for bsl in bsll]
				
	return all_weights

def read_cfg(cfg_fn):
	"""
	"""
	def parse(s):
		""
		if s.isidentifier():
			pass
		elif '.' in s:
			s = float(s)
		else:
			s = int(s)
		return s
	net = []
	current_d = None
	fd = open(cfg_fn,"rt")
	for line in fd:
		line = line.strip()
		if line == "" or line[0] in "#;": continue
		if line[0] == '[':
			assert line[-1] == ']', line
			if current_d is not None:
				net.append(current_d)
			name = line[1:-1]
			assert name.isidentifier()
			current_d = {'' : name}				
		else:
			i = line.index('=')
			name = line[:i].rstrip()
			assert name.isidentifier()
			value = line[i+1:].lstrip()
			if ',' in value:
				value = [parse(s.strip()) for s in value.split(',')]
			else:
				value = parse(value)
			current_d[name] = value
	net.append(current_d)
	fd.close()
	assert net[0][''] == 'net'
	for d in net[1:]:
		assert set(d) <= layer_check_d[d['']]
	return net

layer_check_d = {
	'convolutional': {'', 'activation', 'batch_normalize', 'filters', 'pad', 'size', 'stride'},
	'route': {'', 'layers'},
	'shortcut': {'', 'activation', 'from'},
	'upsample': {'', 'stride'},
	'yolo': {'', 'anchors', 'classes', 'ignore_thresh', 'jitter', 'mask', 'num',  'random', 'truth_thresh'},
	}

def cfg_sizes(cfg, HW=None):
	""
	assert HW is None or len(HW) == 2
	assert cfg[0][''] == 'net'
	assert cfg[0]['channels'] == 3
	if HW is None:
		HW = (cfg[0]['height'], cfg[0]['width'])
	HWC = HW + (3,) # darknet params
	osh = (1,3) + HW                        # CV output shape
	out_channels_vec = [None] * (len(cfg)-1) # liste des osh
	cv_layer_id = 0
	cv_last_layer = cv_FirstLayerName = "data"
	cv_layers = [{'name': cv_FirstLayerName, "os": osh, "bsl":[]}]
	cv_fused_layer_names = []
	layers = cfg[1:] # darknet net->layers
	for cfg_idx, d in enumerate(layers, start=1):
		layers_counter = cfg_idx-1			### 0-based : celui de darknet
		if d[''] == 'convolutional': # conv_x bn_x relu_x
			assert set(d) <= {'', 'activation', 'batch_normalize', 'filters', 'pad', 'size', 'stride'}
			assert d['pad'] == 1
			activation = d.get("activation", "linear")
			assert activation in ('leaky','linear')
			batch_normalize = d.get('batch_normalize',0)
			assert batch_normalize in (0,1)
			filters = d['filters']
			size = d['size']
			assert size in (1,3)
			stride = d['stride']
			assert stride in (1,2)
			# 
			(h,w,c) = HWC
			assert h%stride == w%stride == 0
			out_h = h // stride
			out_w = w // stride
			out_c = filters
			HWC = (out_h, out_w, out_c)
			# blobs
			bl = [("biases", (filters,))]
			if batch_normalize:
				bl.append(("scales",(filters,)))
				bl.append(("rolling_mean",(filters,)))
				bl.append(("rolling_variance",(filters,)))
			bl.append(("weights",(filters*c*size*size,)))
			d['bnl'] = [n for n,_ in bl]
			d['bsl'] = [s for _,s in bl]
			# OpenCV
			cv_layer_name = f"conv_{cv_layer_id}"
			assert osh[2]%stride == 0 and osh[3]%stride == 0
			ish = osh; osh = (ish[0], filters, ish[2]//stride, ish[3]//stride)
			bsl = [(osh[1], ish[1], size, size)]
			if batch_normalize == 0:
				bsl.append((1, osh[1]))
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer], \
			   "isl": [ish], "os": osh, "bsl": bsl, \
			   "bias_term": batch_normalize==0}
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			if batch_normalize:
				cv_layer_name = f"bn_{cv_layer_id}"
				cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer], \
					"isl": [osh], "os": osh, "bsl": [(1,filters)]*4, \
					"has_weight": True, "has_bias": True, "eps": 1E-6}
				cv_layers.append(cv_layer)
				cv_last_layer = cv_layer_name
			if activation == "leaky":
				cv_layer_name = f"relu_{cv_layer_id}"
				cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer], \
					"isl": [osh], "os": osh, "bsl": [], \
					"negative_slope": 0.1}
				cv_layers.append(cv_layer)
				cv_last_layer = cv_layer_name
			cv_layer_id += 1
			cv_fused_layer_names.append(cv_last_layer)
			#
		elif d[''] == 'route': # identity_x ou concat_x
			assert set(d) <= {'', 'layers'}
			layers_vec = d['layers']
			if not isinstance(layers_vec,list):
				layers_vec = [layers_vec]
			assert all(l != 0 for l in layers_vec)
			#
			li1 = layers_vec[0]
			assert li1 < 0
			layer1 = layers[li1+layers_counter]
			HWC1 = layer1['HWC_out']
			if len(layers_vec) == 1:
				HWC = HWC1
			else:
				li2 = layers_vec[1]
				assert li2 > 0
				layer2 = layers[li2] ## 0-based
				HWC2 = layer2['HWC_out']
				assert HWC1[:2] == HWC2[:2]
				HWC = HWC1[:2] + (HWC1[2]+HWC2[2],)
			# 
			layers_vec_abs = [l if l >= 0 else (l + layers_counter) for l in layers_vec]
			isl = [out_channels_vec[l] for l in layers_vec_abs]
			assert all(sh[2:] == isl[0][2:] for sh in isl[1:]), (cfg_idx, cv_layer_id, isl)
			osh = (isl[0][0], sum(sh[1] for sh in isl)) + isl[0][2:]
			# OpenCV
			kind = 'identity' if len(layers_vec) == 1 else 'concat'
			cv_layer_name = f"{kind}_{cv_layer_id}"
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, \
			   "bottom_indexes": [cv_fused_layer_names[l] for l in layers_vec_abs], \
			   "isl": isl, "os": osh, "bsl": [] }
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			cv_layer_id += 1
			cv_fused_layer_names.append(cv_last_layer)
			#
		elif d[''] == 'shortcut': # shortcut_x (cv Eltwise/add)
			assert set(d) <= {'', 'activation', 'from'}
			from_ = d['from']
			assert from_ < 0
			activation = d["activation"]
			assert activation == 'linear'
			# 
			from_layer = layers[from_+layers_counter]
			from_HWC = from_layer['HWC_out']
			assert from_HWC == HWC
			# OpenCV
			from_abs = from_ + layers_counter if from_ < 0 else from_
			ish2 = out_channels_vec[from_abs]
			assert osh == ish2
			isl = [osh,ish2]
			cv_layer_name = f"shortcut_{cv_layer_id}"  #### ATTENTION : cv Eltwise
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, \
			   "bottom_indexes": [cv_last_layer, cv_fused_layer_names[from_abs]], \
			   "isl": isl, "os": osh, "bsl": [], \
			   "op": "sum"}
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			cv_layer_id += 1
			cv_fused_layer_names.append(cv_last_layer)
			#
		elif d[''] == 'upsample': # cv Resize
			assert set(d) <= {'', 'stride'}
			stride = d['stride']
			assert stride == 2
			#
			HWC = (HWC[0]*stride, HWC[1]*stride, HWC[2])
			#
			isl = [osh]; osh = osh[:2] + (osh[2]*stride, osh[3]*stride)
			# OpenCV
			cv_layer_name = f"upsample_{cv_layer_id}"  #### ATTENTION : cv Resize
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer], \
				"isl": isl, "os": osh, "bsl": [], \
				"zoom_factor": stride, "interpolation": "nearest"}
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			cv_layer_id += 1
			cv_fused_layer_names.append(cv_last_layer)
			#
		elif d[''] == 'yolo': # conv_x permute_x+1 yolo_x+1 (cv Region)
			assert set(d) <= {'', 'anchors', 'classes', 'ignore_thresh', 'jitter', 'mask', 'num',  'random', 'truth_thresh'}
			anchors = d['anchors']
			classes = d['classes']
			ignore_thresh = d['ignore_thresh']
			jitter = d['jitter']
			mask = d['mask']
			num_of_anchors = d['num']
			random = d['random']
			truth_thresh = d['truth_thresh']
			assert classes > 0 and (num_of_anchors * 2) == len(anchors)
			perm = [0, 2, 3, 1]
			numAnchors = len(mask)
			usedAnchors = []
			for m in mask:
				usedAnchors.extend(anchors[m*2:m*2+2])
			blob = np.array([usedAnchors], dtype=np.float32) # shape : (1,6)
			#
			(h,w,c) = HWC
			assert c==255
			HWC = (3*h*w, 85) # c=1 implicite
			d["bsl"] = [blob.shape]
			d["blobs"] = [blob]
			# OpenCV
			cv_layer_name = f"permute_{cv_layer_id}"
			isl = [osh]; osh = tuple(osh[p] for p in perm)
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer], \
				"isl": isl, "os": osh, "bsl": [], \
				"order": perm}
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			cv_layer_name = f"yolo_{cv_layer_id}" ### ATTENTION: cv Region
			ish1 = osh; ish2 = (1,3) + HW; isl = [ish1,ish2]
			osh = (3*ish1[1]*ish1[2], 85)
			cv_layer = {'name': cv_layer_name, 'cfg_idx': cfg_idx, "bottom_indexes": [cv_last_layer, cv_FirstLayerName], \
				"isl": isl, "os": osh, "bsl": [blob.shape], \
				"classes": classes, "anchors": numAnchors, "logistic": True, "blobs": [blob]}
			cv_layers.append(cv_layer)
			cv_last_layer = cv_layer_name
			cv_layer_id += 1
			cv_fused_layer_names.append(cv_last_layer)
			#
		else:
			assert False
		d['HWC_out'] = HWC
		print(layers_counter, d[''], HWC)
		out_channels_vec[layers_counter] = osh
	return layers, cv_layers

def cv_layers_dump(cv_layers):
	""
	txt = ''
	for li,l in enumerate(cv_layers):
		ln = l['name']
		lt = l.get('type','???')
		bsh_l = l['bsl']
		ish_l = l.get('isl',[])
		osh_l = [l['os']]
		summary = [ln, lt, bsh_l, ish_l, osh_l]
		txt += f'{li} {ln} {lt}\n\tblobs  : {summary[2]}\n\tinputs : {summary[3]}\n\toutput : {summary[4][0]}\n'
	return txt

def blob_shapes(layers):
	""
	bsll = [l.get('bsl',[]) if l['']!='yolo' else [] for l in layers]
	return bsll

def cv_blob_shapes(cv_layers):
	"les yolos sont exclus"
	bsll = [l['bsl'] if not l['name'].startswith('yolo') else [] for l in cv_layers]
	return bsll
	
def blob_sz(bsll):
	""
	bsz = 0
	for bsl in bsll:
		bsz += sum(prod(bs) for bs in bsl)
	return bsz

def update_yolos_weights(layers, bsll, weights):
	""
	assert len(layers) == len(bsll) == len(weights)
	for i,(l,bsl,bl) in enumerate(zip(layers,bsll,weights)):
		if l.get('','') == 'yolo' or \
		   l.get("name","").startswith('yolo'): # darknet puis OpenCV
			assert bsl == bl == []
			bsl.extend(l['bsl'])
			bl.extend(l['blobs'])

############ image.c #############

f32 = np.float32

"""
// layout IDENTIQUE DARKNET / NP : juste l'invertion des coord :
// dk(w,h,c) == np(c,h,w)
// dk(x,y,c) == np(c,y,x)
static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
"""
def resize_image(im,w,h):
	""
	im_c,im_h,im_w = im.shape
	resized = np.zeros((im_c,h,w), np.float32)
	part = np.zeros((im_c,im_h,w), np.float32)
	w_scale = np.divide(im_w - 1 ,w - 1, dtype=np.float32)
	h_scale = np.divide(im_h - 1 ,h - 1, dtype=np.float32)
	for k in range(im_c):
		for r in range(im_h):
			for c in range(w):
				if c == w-1 or im_w == 1:
					val = im[k,r,im_w-1]
				else:
					sx = f32(c)*w_scale # WARNING : float64
					ix = int(sx)
					dx = sx-f32(ix)
					assert 0 <= dx < 1, dx
					val = (f32(1)-dx)*im[k,r,ix] + dx*im[k,r,ix+1]
				part[k,r,c] = val
	for k in range(im_c):
		for r in range(h):
			sy = f32(r)*h_scale
			iy = int(sy)
			dy = sy - f32(iy)
			assert 0 <= dy < 1, dy
			for c in range(w):
				resized[k,r,c] = (f32(1)-dy) * part[k,iy,c]
			if r == h-1 or im_h == 1: continue
			for c in range(w):
				resized[k,r,c] += dy * part[k,iy+1,c]
	return resized

def embed_image(resized, boxed, dw, dh):
	""
	pass

def letterbox_image(im,w,h):
	""
	[c, o_h, o_w] = im.shape
	if w/o_w < h/o_h:
		new_w = w
		new_h = (o_h * w) // o_w
	else:
		new_h = h
		new_w = (o_w * h) // o_h 
	resized = resize_image(im, new_w, new_h)
	boxed = mp.ones((c,h,w), np.float32) * np.float32(0.5)
	embed_image(resized, boxed, (w-new_w)//2, (h-new_h)//2)
	return boxed

####################################

if __name__ == "__main__":
	
	import imageio
	file = 'horses.jpg' # 512*773
	# RGB
	image_io = imageio.imread(file)
	assert image_io.shape[2] == 3 and image_io.dtype == np.uint8
	if False:
		import cv2 as cv
		# BGR
		image_cv = cv.imread(file)
		assert image_cv.shape == image_io.shape and image_cv.dtype == image_io.dtype
		for i in range(3):
			assert all(image_io[:,:,2-i].flatten() == image_cv[:,:,i].flatten())
	# RGB
	image_dk = np.empty((3,)+image_io.shape[:2], np.float32)
	for i in range(3):
		# image_dk[i,:,:] = np.float32(image_io[:,:,i]) / 255
		image_dk[i,:,:] = np.divide(image_io[:,:,i], 255, dtype=np.float32)
	foo = letterbox_image(image_dk, 608, 608)
	
	
	
	
	cfg = read_cfg('yolov3.cfg')
	HW = (416,416)
	HW = None
	layers, cv_layers = cfg_sizes(cfg, HW)
	# weights = read_weights('yolov3.weights') # 62001757
	bsll = blob_shapes(layers)
	cv_bsll = cv_blob_shapes(cv_layers)
	# bsz = blob_sz(bsll)
	# txt = cv_layers_dump(cv_layers)
	weights = read_weights('yolov3.weights', bsll)
	update_yolos_weights(layers, bsll, weights)
	
	# test avec opencv.dnn
	
	try:
		import cv2 as cv
		cv_ok = True
	except ModuleNotFoundError:
		cv_ok = False
	if False:
		print('test opencv')
		net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
		lnames = net.getLayerNames()
		lnames = ['_input']+lnames # _input
		ll = [net.getLayer(i) for i in range(len(lnames))]
		assert all(l.name == n for l,n in zip(ll,lnames))
		bll_cv = [l.blobs for l in ll]
		bsll_cv = [[b.shape for b in bl] for bl in bll_cv]
		assert bsll_cv == bsll
		for i,(bl_cv,bl) in enumerate(zip(bll_cv, weights)):
			assert len(bl_cv) == len(bl)
			for j, (b_cv,b) in enumerate(zip(bl_cv, bl)):
				if not all(b_cv.flatten() == b.flatten()):
					assert False, (i,j)
		
	
