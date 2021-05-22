# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

import numpy as np, time
import cv2 as cv
import my_cv_dnn

# https://github.com/pjreddie/darknet/blob/master/data/dog.jpg
image, min_confidence, NMS_threshold = "dog.jpg", 0.5, 0.3
image, min_confidence, NMS_threshold = "scream.jpg", 0.5, 0.5
image, min_confidence, NMS_threshold = "horses.jpg", 0.5, 0.3

######################################

# https://github.com/pjreddie/darknet/blob/master/data/coco.names
y_n = "coco.names"
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
y_cfg = "yolov3.cfg"
# https://pjreddie.com/media/files/yolov3.weights
y_weights = "yolov3.weights"


LABELS = open(y_n).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv.dnn.readNetFromDarknet(y_cfg, y_weights)
my_cv_dnn.ReadDarknetFromCfg(y_cfg)
my_cv_dnn.analyze(net)

image = cv.imread(image)
print(image.shape, image.dtype)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
blob1 = my_cv_dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
assert all((blob == blob1).flatten())
net.setInput(blob)
start = time.perf_counter()
layerOutputs = net.forward(ln)
end = time.perf_counter()
print(f"[INFO] YOLO took {end-start} seconds")

assert len(ln) == len(layerOutputs)
boxes = []
confidences = []
classIDs = []
for name, output in zip(ln,layerOutputs):
	for detection in output:
		# extract the class ID and confidence
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > min_confidence:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv.dnn.NMSBoxes(boxes, confidences, min_confidence, NMS_threshold)

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		x, y, w, h = boxes[i]
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

print('done')
cv.imshow("Image", image)
cv.waitKey(0)
