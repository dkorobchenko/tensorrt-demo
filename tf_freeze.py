'''
Prepare TF graph for the inference procedure (mostly required for the further porting to TensorRT)
'''

import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
import model

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

OUTPUT_NAMES = ['net/fc8/BiasAdd'] # ADJUST
INPUT_SIZE = [1, 224, 224, 3] # ADJUST
MODEL_PATH = 'data/checkpoints/vggA_BN/' # ADJUST
FROZEN_GDEF_PATH = 'data/frozen.pb' # ADJUST
EXPORT_FOR_TENSORBOARD = False # ADJUST

### Create TF graph for inference

graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope('net'):
        net_inp = tf.placeholder(tf.float32, INPUT_SIZE, name='input')
        net_out = model.model(net_inp, is_training=False)
    saver = tf.train.Saver()

### Create TF session and load snapshot

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=sess_config)
snapshot_fpath = tf.train.latest_checkpoint(MODEL_PATH)
saver.restore(sess, snapshot_fpath)

### Freeze graph

graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
graphdef_frozen = tf.graph_util.convert_variables_to_constants(
    sess, graphdef_inf, OUTPUT_NAMES)
os.makedirs(os.path.dirname(FROZEN_GDEF_PATH), exist_ok=True)
graph_io.write_graph(graphdef_frozen, './', FROZEN_GDEF_PATH, as_text=False)

### List frozen nodes

print([x.name for x in graphdef_frozen.node])

### Export frozen graph for visualization in Tensorboard

if EXPORT_FOR_TENSORBOARD:
    graph_frozen = tf.Graph()
    with graph_frozen.as_default():
        tf.import_graph_def(graphdef_frozen)
    _=tf.summary.FileWriter('data/checkpoints/vggA_BN_frozen/', graph_frozen)
