import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco
import time
import pandas as pd

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_MAX_DIM = 64
    IMAGE_MIN_DIM = 64
    IMAGE_SHAPE = [64,64,3]

config = InferenceConfig()

coco_model_path = "mask_rcnn_coco.h5"
image_dir =  "../../Data/sample_frames/jackson-town-square/freq002_resori/"
model_dir = "logs"
detect_result = []

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

# Load weights trained on MS-COCO
model.load_weights(coco_model_path, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

start = time.time()

for index in np.arange(0,973101,50):
    image = skimage.io.imread(image_dir+"%d.jpg"%index)
    results = model.detect([image], verbose=1)
    r = results[0]['class_ids']
    for class_id in r:
        if class_id == 3:
            detect_result.append([index, 'car'])

print("Total time:")
print(time.time()-start)

df = pd.DataFrame(detect_result, columns=['frame', 'object_name'])
df.to_csv("../../Data/filtered/jackson-town-square/freq002_res64_mrcnn/jackson-town-square-2017-12-17.csv", index=None)