import numpy as np, cv2 as cv

def blobFromImage(image, scalefactor=1.0, size_WH=None, mean=None, swapRB=False, crop=False, ddepth=cv.CV_32F):
	"""
	WARNING : image est modifie
	size_ et mean_ : par ref
	return blob_
	"""
	assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
	assert ddepth in (cv.CV_32F, cv.CV_8U)
	if ddepth == cv.CV_8U:
		assert scalefactor==1.0 and mean is None
	src_HW = image.shape[:2]
	if size_WH is None:
		size_HW = src_HW
		size_WH = tuple(reversed(size_HW))
	else:
		assert len(size_WH) == 2
		size_HW = tuple(reversed(size_WH))
	if size_HW != src_HW:
		if crop:
			assert False
		else:
			image_1 = cv.resize(image, size_WH, fx=0, fy=0, interpolation = cv.INTER_LINEAR) # dst=image
			assert image_1.shape == size_HW + (3,)
	else:
		image_1 = image
	if image.dtype == np.uint8 and ddepth == cv.CV_32F:
		image_2 = np.float32(image_1)
	else:
		image_2 = image_1
	if swapRB:
		assert mean is None
	mean = np.float32([0,0,0])
	image_2 -= mean
	image_2 *= np.float32(scalefactor)
	blob = np.empty((1,3) + size_HW, dtype = np.float32)
	for ch in range(3):
		blob[0,ch,:,:] = image_2[:,:,(2-ch if swapRB else ch)]
	return blob


def NMSBoxes(boxes, confidences, min_confidence, NMS_threshold):
	"""
	AUCUN ENJEU : NE PAS FAIRE
	"""
	return 0

#######################################################

def ReadDarknetFromCfg(cfg_fn, CV_450 = False):
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
	#
	assert net[-1][''] == 'yolo'
	anchors = net[-1]['anchors']
	assert isinstance(anchors, list) and all(isinstance(x,int) for x in anchors)
	
	net_cfg = net[0]
	layers_cfg = net[1:]
	net = {'net_cfg': net_cfg, 'layers_cfg': layers_cfg}
	assert net_cfg[''] == 'net'
	net['width'] = net_cfg.get('width',416)
	net['height'] = net_cfg.get('height',416)
	net['channels'] = net_cfg.get('channels',3)
	assert  all(net[x] > 0 for x in (('width','height','channels')))
	#####
	# avant CV_450, seul existe tensor_shape[0] sous le nom current_channels
	#####
	tensor_shape = [net['channels'], net['width'], net['height']]
	net['out_channels_vec'] = out_channels_vec = [None] * len(layers_cfg)
	net['layers'] = layers = [] # darknet::LayerParameter list
	
	kFirstLayerName = "data"
	# class setLayersParams
	layer_id = 0
	last_layer = kFirstLayerName
	fused_layer_names = []
	# chaque darknet layer pointe vers un dnn layer
	for layers_counter, lcfg in enumerate(layers_cfg):
		lt = lcfg['']
		if lt == "convolutional":
			kernel_size = lcfg.get('size', -1)
			pad = lcfg.get('pad',0)
			stride = lcfg.get('stride', 1)
			filters = lcfg.get('filters',-1)
			batch_normalize = lcfg.get('batch_normalize',0) == 1
			flipped = lcfg.get('flipped', 0)
			assert flipped==0, "Transpose the convolutional weights is not implemented"
			assert stride > 0 and kernel_size > 0 and filters > 0
			assert tensor_shape[0] > 0
			if CV_450:
				padding = lcfg.get('padding', 0)
				if pad:
					padding = kernel_size/2
				groups = lcfg.get('groups', 1)
				assert tensor_shape[0] % groups == 0
			else:
				activation = lcfg.get("activation", "linear")
				assert activation in ("linear","leaky"), "Unsupported activation: " + activation
				if kernel_size < 3:
					pad = 0
			# CV_450: setParams.setConvolution(kernel_size, padding, stride, filters, tensor_shape[0], groups, batch_normalize);
			# CV_401:             ...                       pad            ...           activation == "leaky"
			dnn_lp = {
				"name": "Convolution-name",
				"type" : "Convolution",
				"kernel_size": kernel_size,
				"pad": padding if CV_450 else pad,
				"stride": stride,
				"bias_term": False,
				"num_output": filters }
			if not batch_normalize:
				dnn_lp["bias_term"] = True
			layer_name = f"conv_{layer_id}"
			dark_lp = {
				"layer_name": layer_name,
				"layer_type": dnn_lp["type"],
				"layerParams": dnn_lp,
				"bottom_indexes": [last_layer] }
			last_layer = layer_name
			layers.append(dark_lp)
			if batch_normalize:
				# setBatchNorm();
				dnn_lp = {
					"name": "BatchNorm-name",
					"type" : "BatchNorm",
					"has_weight": True,
					"has_bias": True,
					"eps": 1E-6 }
				layer_name = f"bn_{layer_id}"
				dark_lp = {
					"layer_name": layer_name,
					"layer_type": dnn_lp["type"],
					"layerParams": dnn_lp,
					"bottom_indexes": [last_layer] }
				last_layer = layer_name
				layers.append(dark_lp)
			if (not CV_450) and activation == "leaky":
				dnn_lp = {
					"name": "ReLU-name",
					"type" : "ReLU",
					"negative_slope": 0.1}
				layer_name = f"relu_{layer_id}"
				dark_lp = {
					"layer_name": layer_name,
					"layer_type": dnn_lp["type"],
					"layerParams": dnn_lp,
					"bottom_indexes": [last_layer] }
				last_layer = layer_name
				layers.append(dark_lp)
			layer_id += 1
			fused_layer_names.append(last_layer)
			# end setConvolution
			tensor_shape[0] = filters
			if CV_450:
				tensor_shape[1] = (tensor_shape[1] - kernel_size + 2 * padding) / stride + 1
				tensor_shape[2] = (tensor_shape[2] - kernel_size + 2 * padding) / stride + 1
		elif lt == "connected":
			assert False
		elif lt == "maxpool":
			assert False
		elif lt == "avgpool":
			assert False
		elif lt == "softmax":
			assert False
		elif lt == "route":
			layers_vec = lcfg["layers"]
			if not isinstance(layers_vec, list):
				layers_vec = [layers_vec]
			if CV_450:
				groups = lcfg.get("groups", 1)
			tensor_shape[0] = 0
			for k,l in enumerate(layers_vec):
				layers_vec[k] = l if l >= 0 else (l + layers_counter) # CV_401 : > au lieu de >=
				tensor_shape[0] += out_channels_vec[layers_vec[k]]
			if CV_450 and groups > 1:
				assert False
			else:
				if len(layers_vec) == 1:
					# setParams.setIdentity(layers_vec.at(0));
					dnn_lp = {
						"name": "Identity-name",
						"type" : "Identity" }
					layer_name = f"identity_{layer_id}"
					dark_lp = {
						"layer_name": layer_name,
						"layer_type": dnn_lp["type"],
						"layerParams": dnn_lp,
						"bottom_indexes": [fused_layer_names[layers_vec[0]]] }
					last_layer = layer_name
					layers.append(dark_lp)
					layer_id += 1
					fused_layer_names.append(last_layer)
				else:
					# setParams.setConcat(layers_vec.size(), layers_vec.data());
					dnn_lp = {
						"name": "Concat-name",
						"type" : "Concat",
						"axis": 1 }
					layer_name = f"concat_{layer_id}"
					dark_lp = {
						"layer_name": layer_name,
						"layer_type": dnn_lp["type"],
						"layerParams": dnn_lp,
						"bottom_indexes": [fused_layer_names[l] for l in layers_vec] }
					last_layer = layer_name
					layers.append(dark_lp)
					layer_id += 1
					fused_layer_names.append(last_layer)
		elif lt in ("dropout", "cost"):
			assert False
		elif lt == "reorg":
			assert False
		elif lt == "region":
			assert False
		elif lt == "shortcut":
			from_ = lcfg["from"]
			if CV_450:
				alpha = lcfg.get("alpha", 1)
				beta = lcfg.get("beta", 0)
				assert beta == 0, "Non-zero beta"
			else:
				assert from_ < 0
			from_ = from_ + layers_counter if from_ < 0 else from_
			if not CV_450:
				tensor_shape[0] = out_channels_vec[from_]
			# setParams.setShortcut(from, alpha);
			dnn_lp = {
				'name': "Shortcut-name",
				'type': "Eltwise",
				"op": "sum" }
			if CV_450:
				dnn_lp["output_channels_mode"] = "input_0_truncate"
				if alpha != 1:
					assert False
			layer_name = f"shortcut_{layer_id}"
			dark_lp = {
				'layer_name': layer_name,
				'layer_type': dnn_lp['type'],
				'layerParams': dnn_lp,
				"bottom_indexes": [last_layer, fused_layer_names[from_]] if CV_450 \
							else  [fused_layer_names[from_], last_layer] }
			last_layer = layer_name
			layers.append(dark_lp)
			layer_id += 1
			fused_layer_names.append(last_layer)
			# end setShortcut
		elif lt == "scale_channels":
			assert False
		elif lt == "upsample":
			scaleFactor = lcfg.get("stride", 1)
			# setParams.setUpsample(scaleFactor);
			dnn_lp = {
				'name': "Upsample-name",
				'type': "Resize",
				"zoom_factor": scaleFactor,
				"interpolation": "nearest" }
			layer_name = f"upsample_{layer_id}"
			dark_lp = {
				'layer_name': layer_name,
				'layer_type': dnn_lp['type'],
				'layerParams': dnn_lp,
				"bottom_indexes": [last_layer] }
			last_layer = layer_name
			layers.append(dark_lp)
			layer_id += 1
			fused_layer_names.append(last_layer)
			# end setUp...
			if CV_450:
				tensor_shape[1] *= scaleFactor
				tensor_shape[2] *= scaleFactor
		elif lt == "yolo":
			classes = lcfg.get("classes", -1)
			num_of_anchors = lcfg.get("num", -1)
			if CV_450:
				thresh = lcfg.get("thresh", 0.2)
				nms_threshold = lcfg.get("nms_threshold", 0.0)
				scale_x_y = lcfg.get("scale_x_y", 1.0)
			anchors_vec = lcfg["anchors"]
			mask_vec = lcfg["mask"]
			assert classes > 0 and num_of_anchors > 0 and (num_of_anchors * 2) == len(anchors_vec)
			# setParams.setPermute(false);
			dnn_lp = {
				'name': "Permute-name",
				'type': "Permute",
				"order": [0, 2, 3, 1] }
			layer_name = f"permute_{layer_id}"
			dark_lp = {
				'layer_name': layer_name,
				'layer_type': dnn_lp['type'],
				'layerParams': dnn_lp,
				"bottom_indexes": [last_layer] }
			last_layer = layer_name
			layers.append(dark_lp)
			if False:
				layer_id += 1
				fused_layer_names.append(last_layer)
			# setParams.setYolo(classes, mask_vec, anchors_vec, thresh, nms_threshold, scale_x_y);
			numAnchors = len(mask_vec)
			usedAnchors = []
			for m in mask_vec:
				usedAnchors.extend(anchors[m*2:m*2+2])
			dnn_lp = {
				'name': "Region-name",
				'type': "Region",
				"classes": classes,
				"anchors": numAnchors,
				"logistic": True,
				"blobs": [usedAnchors] }
			if CV_450:
				dnn_lp.update({
				"thresh": thresh,
				"nms_threshold": nms_threshold,
				"scale_x_y": scale_x_y })
			layer_name = f"yolo_{layer_id}"
			dark_lp = {
				'layer_name': layer_name,
				'layer_type': dnn_lp['type'],
				'layerParams': dnn_lp,
				"bottom_indexes": [last_layer, kFirstLayerName] }
			last_layer = layer_name
			layers.append(dark_lp)
			layer_id += 1
			fused_layer_names.append(last_layer)
		else:
			assert False, "Unknown layer type: "+lt
		if CV_450:
			activation = lcfg.get("activation", "linear")
			if activation != "linear":
				# setParams.setActivation(activation);
				dnn_lp = {}
				if activation == "relu":
					dnn_lp["type"] = "ReLU"
				elif activation == "leaky":
					dnn_lp["negative_slope"] = 0.1
					dnn_lp["type"] = "ReLU"
				elif activation == "swish":
					dnn_lp["type"] = "Swish"
				elif activation == "mish":
					dnn_lp["type"] = "Mish"
				elif activation == "logistic":
					dnn_lp["type"] = "Sigmoid"
				else:
					assert False, "Unsupported activation: " + activation
				layer_name = f"{activation}_{layer_id}"
				dark_lp = {
					'layer_name': layer_name,
					'layer_type': dnn_lp['type'],
					'layerParams': dnn_lp,
					"bottom_indexes": [last_layer] }
				last_layer = layer_name
				layers.append(dark_lp)
				fused_layer_names[-1] = last_layer
			# end setActivation
		out_channels_vec[layers_counter] = tensor_shape[0]
		
	return net
	

#######################################################

def BatchNorm_shapes(bsh_l, ish_l, params={}):
	""
	assert len(bsh_l) == 4, bsh_l
	[ish] = ish_l
	assert all(bsh==(1, ish[1]) for bsh in bsh_l)
	return ish

def Concat_shapes(bsh_l, ish_l, params={}):
	""
	assert bsh_l == []
	i1,i2 = ish_l
	assert i1[0] == i2[0] and i1[2:] == i2[2:]
	return (i1[0], i1[1]+i2[1]) + i1[2:]

def Convolution_shapes(bsh_l, ish_l, params={'stride':1}):
	""
	assert len(bsh_l) in (1,2)
	bsh = bsh_l[0]
	assert bsh[2] == bsh[3]
	assert bsh[2] in (1,3), bsh[2]
	if len(bsh_l) == 2:
		# blobs  : [(255, 256, 1, 1), (1, 255)]
		assert bsh_l[1] == (1, bsh[0])
	[ish] = ish_l
	assert ish[0] ==1 and ish[1] == bsh[1]
	stride = params.get('stride',1)
	return (1, bsh[0]) + (ish[2]/stride, ish[3]/stride)

def Convolution_k1(b,bias,inp,o):
	""
	bs = b.shape
	os = o.shape
	plan = np.empty(os[2:], dtype=o.dtype)
	for i in range(bs[0]):
		plan[:,:] = 0 if bias is None else bias[0,i]
		for j in range(bs[1]):
			plan += b[i,j,0,0] * inp[0,j,:,:]
		o[0,i,:,:] = plan

def Convolution_k3(b,bias,inp,o):
	""
	print("not implemented")

def Convolution_forward(bl,il, stride=1):
	"""
7 conv_2 Convolution
	blobs  : [(32, 64, 1, 1)]
	inputs : [[1, 64, 208, 208]]
	output : [1, 32, 208, 208]
	
	[b7] = net.getLayer('conv_2').blobs
	xx = Convolution_forward([b7],[y6])
	np.max(np.abs(xx.flatten()-y7.flatten())) -> 3.8146973e-06
	
252 conv_105 Convolution
	blobs  : [(255, 256, 1, 1), (1, 255)]
	inputs : [[1, 256, 52, 52]]
	output : [1, 255, 52, 52]
	
	bl = net.getLayer('conv_105').blobs pour bias
	zz = Convolution_forward(bl,[xx])
	np.max(np.abs(zz.flatten()-yy.flatten())) -> 
	"""
	b = bl[0]
	bs = b.shape
	if len(b)==2:
		bias = bl[1]
		assert bias.shape == (1, bs[0])
	else:
		bias = None
	[inp] = il
	inps = inp.shape
	assert bs[1] == inps[1] and inps[0] == 1
	os = (1, bs[0]) + inps[2:]
	o = np.empty(os, dtype = inp.dtype)
	if bs[2:] == (1,1):
		Convolution_k1(b,bias,inp,o)
	elif bs[2:] == (3,3):
		Convolution_k3(b,bias,inp,o)
	else:
		assert False, "kernel size not implemented"
	return o	

def ReLU_shapes(bsh_l, ish_l, params={}):
	""
	assert bsh_l == []
	[ish] = ish_l
	return ish
	

layer_d = { # fils de Layer, sauf precision ; cf all_layers.hpp
'': None,
'BatchNorm': [BatchNorm_shapes], # Activation
'Concat': [Concat_shapes],
'Convolution': [Convolution_shapes], # BaseConvolution
'Eltwise': None,
'Identity': None, # ?
'Permute': None,
'Region': None,
'ReLU': [ReLU_shapes],   # Activation
'Resize': None,
}

def analyze(net, inputShape):
	"""
dnn_Net methods:
connect
empty
enableFusion
forward
forwardAndRetrieve
getFLOPS
getLayer(int) -> layer
getLayerId(str) -> int
getLayerNames() -> str list
getLayerTypes() -> ['BatchNorm', ... 'Resize', '__NetInputLayer__']
getLayersCount(str) -> int
getLayersShapes([1,3,lin,col]) -> list*list*list
getMemoryConsumption
getParam(int,int) -> blob
getPerfProfile
getUnconnectedOutLayers() -> int32 array
getUnconnectedOutLayersNames() -> str list
readFromModelOptimizer
setHalideScheduler
setInput
setInputsNames
setParam
setPreferableBackend
setPreferableTarget

dnn_Layer methods:
finalize
outputNameToIndex
run
blobs
name
preferableTarget
type
	"""
	sl = []
	sbig = ''
	ll = []
	#s = set()
	l_input = net.getLayer(0)
	assert  l_input.name == '_input' \
		and net.getLayerId('_input') == 0 \
		and l_input.type == '' \
		and l_input.blobs == []
	layers_ids, in_shapes, out_shapes = net.getLayersShapes(inputShape)
	lnl = ['_input'] + net.getLayerNames()
	assert len(layers_ids) == len(in_shapes) == len(out_shapes) == len(lnl)
	assert all(layers_ids[:,0] == range(len(lnl)))
	for li, ln in enumerate(lnl):
		assert net.getLayerId(ln) == li
		layer = net.getLayer(li)
		assert layer.name == ln
		lt = layer.type
		lt_methods = layer_d[lt]
		bl = layer.blobs
		assert isinstance(bl, list)
		in_s = in_shapes[li]
		assert isinstance(in_s , list) and all(sh.shape in [(4,1),(2,1)] for sh in in_s ), in_s
		assert len(in_s) in (1,2)
		out_s = out_shapes[li]
		assert isinstance(out_s, list) and all(sh.shape in [(4,1),(2,1)] for sh in out_s), out_s
		assert len(out_s) == 1, (lt, out_s)
		bsh_l = [b.shape for b in bl]
		ish_l = [tuple(x.flatten().tolist()) for x in in_s]
		osh_l = [tuple(x.flatten().tolist()) for x in out_s]
		if lt_methods:
			osh = lt_methods[0](bsh_l, ish_l)
			if osh != osh_l[0]:
				if lt == 'Convolution':
					osh = lt_methods[0](bsh_l, ish_l,{'stride':2})
					assert osh == osh_l[0]
				else:
					assert False
		#
		summary = [ln, lt, bsh_l, ish_l, osh_l]
		txt = f'{li} {ln} {lt}\n\tblobs  : {summary[2]}\n\tinputs : {summary[3]}\n\toutput : {summary[4][0]}\n'
		print(txt)
		#print(li, ln, lt)
		#print('\tblobs  : ', summary[2])
		#print('\tinputs : ', summary[3])
		#print('\toutput : ', summary[4][0])
		ll.append(summary)
		sl.append(txt)
		sbig += txt
	return ll
	
if __name__ == "__main__":
	net_ = ReadDarknetFromCfg('yolov3.cfg')