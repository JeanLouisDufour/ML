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

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])  

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))     
    
    def reset(self):
        self.offset = 0

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
	assert len(HW) == 2
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
	for cfg_idx, d in enumerate(cfg[1:], start=1):
		layers_counter = cfg_idx-1
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
			d['HWC_out'] = HWC
			# blobs
			d['blobs'] = bl = [("biases", (filters,))]
			if batch_normalize:
				bl.append(("scales",(filters,)))
				bl.append(("rolling_mean",(filters,)))
				bl.append(("rolling_variance",(filters,)))
			bl.append(("weights",(filters*c*size*size,)))
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
			layer1 = cfg[li1+cfg_idx]
			HWC1 = layer1['HWC_out']
			if len(layers_vec) == 1:
				HWC = HWC1
			else:
				li2 = layers_vec[1]
				assert li2 > 0
				layer2 = cfg[li2] ## 1-based ????
				HWC2 = layer2['HWC_out']
				assert HWC1[1:] == HWC2[1:]
				HWC = (HWC1[0]+HWC2[0],) + HWC1[1:]
			d['HWC_out'] = HWC
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
			from_layer = cfg[from_+cfg_idx]
			from_HWC = from_layer['HWC_out']
			assert from_HWC == HWC
			d['HWC_out'] = HWC
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
			TBD
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
			blob = np.array([usedAnchors], dtype=np.float32) # shape : (1,6)
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
		out_channels_vec[layers_counter] = osh
	return cv_layers

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

def blob_shapes(cfg):
	""
	bsll = [[sh for _,sh in l.get('blobs', [])] for l in cfg[1:]]
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

def update_yolos_weights(cv_layers, bsll, weights):
	""
	assert len(cv_layers) == len(bsll) == len(weights)
	for i,(l,bsl,bl) in enumerate(zip(cv_layers,bsll,weights)):
		if l["name"].startswith('yolo'):
			assert bsl == bl == []
			bsl.extend(l['bsl'])
			bl.extend(l['blobs'])

if __name__ == "__main__":
	cfg = read_cfg('yolov3.cfg')
	cv_layers = cfg_sizes(cfg, (416,416))
	# weights = read_weights('yolov3.weights') # 62001757
	bsll = blob_shapes(cfg)
	cv_bsll = cv_blob_shapes(cv_layers)
	# bsz = blob_sz(bsll)
	# txt = cv_layers_dump(cv_layers)
	weights = read_weights('yolov3.weights', bsll)
	update_yolos_weights(cv_layers, bsll, weights)
	
	# test avec opencv.dnn
	
	try:
		import cv2 as cv, foo
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
		
	except ModuleNotFoundError:
		print('opencv needed for more tests')
