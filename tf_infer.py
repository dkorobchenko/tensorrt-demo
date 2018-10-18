'''
Perform inference by means of TensorFlow.
There are two modes: "TF" and "TRT", which mean regular TF graph and TF graph with TRT optimizations correspondingly.
The second will be available after the "TensorFlow optimize" step.
'''

import sys
import numpy as np
import imageio
import tensorflow as tf
import utils

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

INPUT_NODE = 'net/input:0' # ADJUST
OUTPUT_NODE = 'net/fc8/BiasAdd:0' # ADJUST
CLASSES = ['Cat', 'Dog'] # ADJUST
CROP_SIZE = (224, 224) # ADJUST
MEASURE_TIME = True # ADJUST
CALC_VAL_ACCURACY = True # ADJUST

### Select graph: Regular TensorFlow or TensorFlow+TensorRT

if len(sys.argv) > 1 and sys.argv[1] == 'TF':
    print('Using regular frozen TF graph')
    GDEF_PATH = 'data/frozen.pb' # ADJUST
elif len(sys.argv) > 1 and sys.argv[1] == 'TRT':
    print('Using TF graph with TensorRT optimizations')
    GDEF_PATH = 'data/frozen_trt.pb' # ADJUST
    import tensorflow.contrib.tensorrt # Required to init TRTEngineOp
else:
    print('Usage: python tf_infer.py <TRT|TF>')
    sys.exit()

### Load frozen graph and create TF session

graph_def = tf.GraphDef()
with tf.gfile.GFile(GDEF_PATH, "rb") as f:
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    net_inp, net_out = tf.import_graph_def(
        graph_def, return_elements=[INPUT_NODE, OUTPUT_NODE])
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=sess_config)

### Load and prepare input image

def prepare_image(img_in, crop_size):
    img = utils.resize_and_crop(img_in, crop_size)
    img = img.astype(np.float32)
    img = img[None, ...]
    return img

INPUT_IMAGE_PATH = 'img/cat.png' # ADJUST
img = imageio.imread(INPUT_IMAGE_PATH, pilmode='RGB')
img = prepare_image(img, CROP_SIZE)

### Run inference

out = sess.run(net_out, feed_dict={net_inp: img})
print('Input: {}'.format(INPUT_IMAGE_PATH))
print('Output: {}'.format(out))
print('Prediction: {}'.format(CLASSES[np.argmax(out)]))

### Measure execution time

if MEASURE_TIME:
    import time
    TIMEIT_N_SKIP = 10 # ADJUST
    TIMEIT_N_RUN = 20 # ADJUST
    imfer_time_arr = []
    for _ in range(TIMEIT_N_SKIP):
        out = sess.run(net_out, feed_dict={net_inp: img})
    for _ in range(TIMEIT_N_RUN):
        time_start = time.time()
        out = sess.run(net_out, feed_dict={net_inp: img})
        imfer_time_arr.append(time.time() - time_start)
    print('Inference time: {:.3f} +- {:.3f} ms (Avg over {} runs, {} skipped)'.format(
        np.mean(imfer_time_arr)*1000.,
        np.std(imfer_time_arr)*1000.,
        TIMEIT_N_RUN, TIMEIT_N_SKIP))

### Calculate ImageNet validation accuracy

if CALC_VAL_ACCURACY:
    import data_provider
    image_list, label_list = data_provider.prepare_sample_list(
        '/imagenet/val/','/imagenet/val.txt', classes=[281, 239])
    correct = 0
    for img_fpath, label in zip(image_list, label_list):
        img = imageio.imread(img_fpath, pilmode='RGB')
        img = prepare_image(img, CROP_SIZE)
        out = sess.run(net_out, feed_dict={net_inp: img})
        if np.argmax(out) == label:
            correct += 1
    accuracy = float(correct) / len(image_list)
    print('ImageNet validation accuracy: {}'.format(accuracy))
