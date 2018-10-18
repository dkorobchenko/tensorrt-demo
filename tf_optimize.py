'''
Use TensorRT+TensorFlow to build a new TF graph with optimized TRT-based subgraphs
'''

import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

FROZEN_GDEF_PATH = 'data/frozen.pb' # ADJUST
TRT_GDEF_PATH = 'data/frozen_trt.pb' # ADJUST
OUTPUT_NODE = 'net/fc8/BiasAdd' # ADJUST
MAX_BATCH_SIZE = 1 # ADJUST
MAX_WORKSPACE = 1 << 32 # ADJUST
DATA_TYPE = 'FP16' # ADJUST # 'FP16' | 'FP32'
EXPORT_FOR_TENSORBOARD = False # ADJUST

### Load frozen TF graph

graphdef_frozen = tf.GraphDef()
with tf.gfile.GFile(FROZEN_GDEF_PATH, "rb") as f:
    graphdef_frozen.ParseFromString(f.read())

### Build new graph with optimized TensorRT nodes

graphdef_trt = trt.create_inference_graph(
    input_graph_def=graphdef_frozen,
    outputs=[OUTPUT_NODE],
    max_batch_size=MAX_BATCH_SIZE,
    max_workspace_size_bytes=MAX_WORKSPACE,
    precision_mode=DATA_TYPE)

### Save new TensorRT graph

os.makedirs(os.path.dirname(TRT_GDEF_PATH), exist_ok=True)
graph_io.write_graph(graphdef_trt, './', TRT_GDEF_PATH, as_text=False)

### List frozen nodes

print([x.name for x in graphdef_trt.node])

### Export new graph for visualization in Tensorboard

if EXPORT_FOR_TENSORBOARD:
    graph_trt = tf.Graph()
    with graph_trt.as_default():
        tf.import_graph_def(graphdef_trt)
    _=tf.summary.FileWriter('data/checkpoints/vggA_BN_frozen_trt/', graph_trt)
