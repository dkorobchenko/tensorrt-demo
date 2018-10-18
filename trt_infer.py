'''
Inference by means of TensorRT
'''

import numpy as np
import imageio
import pycuda.driver as cuda
import pycuda.autoinit # For automatic creation and cleanup of CUDA context
import tensorrt as trt
import utils

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

ENGINE_PATH = 'data/engine.plan' # ADJUST
CLASSES = ['Cat', 'Dog'] # ADJUST
CROP_SIZE = (224, 224) # ADJUST
INPUT_DATA_TYPE = np.float32 # ADJUST
MEASURE_TIME = True # ADJUST
CALC_VAL_ACCURACY = True # ADJUST

### Load TensorRT engine

trt_logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(trt_logger)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

### Prepare TRT execution context, CUDA stream and necessary buffers

context = engine.create_execution_context()
stream = cuda.Stream()
host_in = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=INPUT_DATA_TYPE)
host_out = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=INPUT_DATA_TYPE)
devide_in = cuda.mem_alloc(host_in.nbytes)
devide_out = cuda.mem_alloc(host_out.nbytes)

### Load and prepare input image

def prepare_image(img_in, crop_size):
    img = utils.resize_and_crop(img_in, crop_size)
    img = img.astype(INPUT_DATA_TYPE)
    img = img.transpose(2, 0, 1) # to CHW
    return img

INPUT_IMAGE_PATH = 'img/cat.png' # ADJUST
img = imageio.imread(INPUT_IMAGE_PATH, pilmode='RGB')
img = prepare_image(img, CROP_SIZE)

### Run inference

def infer(img):
    bindings = [int(devide_in), int(devide_out)]
    np.copyto(host_in, img.ravel())
    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()
    return host_out

out = infer(img)
print('Input : {}'.format(INPUT_IMAGE_PATH))
print('Output: {}'.format(out))
print('Prediction: {}'.format(CLASSES[np.argmax(out)]))

### Measure execution time

if MEASURE_TIME:
    import time
    TIMEIT_N_SKIP = 10 # ADJUST
    TIMEIT_N_RUN = 20 # ADJUST
    imfer_time_arr = []
    for _ in range(TIMEIT_N_SKIP):
        out = infer(img)
    for _ in range(TIMEIT_N_RUN):
        time_start = time.time()
        out = infer(img)
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
        out = infer(img)
        if np.argmax(out) == label:
            correct += 1
    accuracy = float(correct) / len(image_list)
    print('ImageNet validation accuracy: {}'.format(accuracy))
