# -*- coding: utf-8 -*-


import numpy as np
import os
import time

import tensorflow as tf
from skimage import io
io.use_plugin('matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# What model to download.
time1=time.time()
files=0
MODEL_NAME = 'model/ssd_mobilenet_v1_coco_2018_01_28/saved_model/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'
print('path is:',PATH_TO_CKPT)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'walkman.pbtxt')

NUM_CLASSES = 1

tf.reset_default_graph()

        
od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')       
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)        
        
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image).reshape(
      (im_height, im_width, 3)).astype(np.uint8)        
        
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_changshidata/bottle'
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)        
        
textfilename = 'result_changshidata/' + "result.txt"
if not os.path.exists(textfilename):
      f = open(textfilename,'w')
      f.close()
detection_graph = tf.get_default_graph()        
with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      print "image:path is ",image_path
      #new11,new_pic11=image_path.split('/')
      files+=1
      image = Image.open(PATH_TO_TEST_IMAGES_DIR+'/'+image_path)
      img_width,img_height=image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #print("box information is:",boxes[0])
      box_new=[boxes[0][0][1]*img_width,boxes[0][0][0]*img_height,boxes[0][0][3]*img_width,boxes[0][0][2]*img_height]
      f=open(textfilename,"r+")
      f.read()
      f.write(image_path+","+str(box_new[0])+","+str(box_new[1])+","+str(box_new[2])+","+str(box_new[3])+"\n")
     # generate a txt for prediction
      #print("boxes is follow",boxes[0])
      #print("scores is follow",scores)
      #print("classes is follow",classes)
      #print("num_detections is follow",num_detections)
      # Visualization of the results of a detection.
      
      print("box_new is",box_new)
      
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np) 
     
      new_pic='result_changshidata/'+image_path
      plt.savefig(new_pic, format='png')
      plt.close()
f.close()
time2=time.time()
avetime=(time2-time1)/files
print('average time is',avetime)
       
        
        
        
        
        
        
        
        
        
        
        
        
        