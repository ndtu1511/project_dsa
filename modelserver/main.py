import base64
import json
import os
import sys
import time

import cv2 as cv
import numpy as np
import redis

# Connect to Redis server
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

conf_threshold = float(os.environ.get("CONF_THRESHOLD"))
nms_threshold = float(os.environ.get("NMS_THRESHOLD"))
img_size = int(os.environ.get("IMAGE_SIZE"))
class_name = 'coco.names'
classes = None
with open(class_name, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
model_configuration = "yolov3.cfg"
model_weights = "yolov3.weights"


net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def base64_decode_image(image, shape):
    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        image = bytes(image, encoding="utf-8")

    # Convert the string to a NumPy array using the supplied data
    # type and target shape
    image = np.frombuffer(base64.decodestring(image), dtype=np.uint8)
    image = image.reshape(shape)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # Return the decoded image
    return image
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def post_process(width_image, height_image, outs, classes):
    classIds = []
    confidences = []
    boxes = []
    outputs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width_image)
                center_y = int(detection[1] * height_image)
                width = int(detection[2] * width_image)
                height = int(detection[3] * height_image)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        right = left+box[2]
        bottom = top+box[3]
        percent = confidences[i]
        if classes:
            assert(classId < len(classes))
            label = classes[classIds[i]]
        value = {
            'object_number': int(i),
            'label': label,
            'percent': float(percent),
            'xmin': int(left),
            'ymin': int(top),
            'xmax': int(right),
            'ymax': int(bottom)
        }
        outputs.append(value)
    return outputs

def detection_process():
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()
        imageIDs = []
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            imageIDs.append(q["id"])
            break
        if len(imageIDs)>0:
            print('Batch length: ',len(imageIDs))
            for q in queue:
                # Deserialize the object and obtain the input image
                q = json.loads(q.decode("utf-8"))
                image = base64_decode_image(q["image"],(int(q["height"]),int(q["width"]),3))
                blob = cv.dnn.blobFromImage(image, 1/255, (img_size, img_size), [0,0,0], 1, crop=False)
                net.setInput(blob)
                outs = net.forward(get_outputs_names(net))
                output = post_process(q["width"],q["height"],outs,classes)
                db.set(q["id"], json.dumps(output))

        # Sleep for a small amount
        time.sleep(float(os.environ.get("SERVER_SLEEP")))

if __name__ == "__main__":
    detection_process()