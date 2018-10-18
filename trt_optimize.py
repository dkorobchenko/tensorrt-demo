'''
Optimize frozen TF graph and prepare a stand-alone TensorRT inference engine
'''

import tensorrt as trt
import uff

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

FROZEN_GDEF_PATH = 'data/frozen.pb' # ADJUST
ENGINE_PATH = 'data/engine.plan' # ADJUST
INPUT_NODE = 'net/input' # ADJUST
OUTPUT_NODE = 'net/fc8/BiasAdd' # ADJUST
INPUT_SIZE = [3, 224, 224] # ADJUST
MAX_BATCH_SIZE = 1 # ADJUST
MAX_WORKSPACE = 1 << 32 # ADJUST
DATA_TYPE = trt.float16 # ADJUST # float16 | float32

### Convert TF frozen graph to UFF graph

uff_model = uff.from_tensorflow_frozen_model(FROZEN_GDEF_PATH, [OUTPUT_NODE])

### Create TRT model builder

trt_logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(trt_logger)
builder.max_batch_size = MAX_BATCH_SIZE
builder.max_workspace_size = MAX_WORKSPACE
builder.fp16_mode = (DATA_TYPE == trt.float16)

### Create UFF parser

parser = trt.UffParser()
parser.register_input(INPUT_NODE, INPUT_SIZE)
parser.register_output(OUTPUT_NODE)

### Parse UFF graph

network = builder.create_network()
parser.parse_buffer(uff_model, network)

### Build optimized inference engine

engine = builder.build_cuda_engine(network)

### Save inference engine

with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())
