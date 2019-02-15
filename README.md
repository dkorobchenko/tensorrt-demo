
# TensorRT Demo

The demo shows how to build, train and test a ConvNet using TensorFlow and then how to port it to TensorRT for fast inference.

There are two ways to perform the TensorRT optimization:
 - Build a stand-alone TensorRT engine
 - Use TensorRT+TensorFlow to build a new TF graph with optimized TRT-based subgraphs

For each step there is a `py` script and a Jupyter Notebook `ipynb`

Parameters, which could be adjusted, are marked with `# ADJUST` comment

## TensorFlow part

*Requirements:* It is recommended to run everything inside `tensorflow` docker container (see docker details below)

### TensorFlow train

Build and train a ConvNet using Tensorflow

Jupyter version: `Tensorflow_train.ipynb`

Python version: `tf_train.py`

### TensorFlow graph freeze

Prepare TF graph for the inference procedure (mostly required for the further porting to TensorRT)

Jupyter version: `Tensorflow_freeze.ipynb`

Python version: `tf_freeze.py`

### TensorFlow inference

Perform inference by means of TensorFlow. There are two modes: `TF` and `TRT`, which mean regular TF graph and TF graph with TRT optimizations correspondingly. The second will be available after the `TensorFlow optimize` step.

Jupyter version: `Tensorflow_infer.ipynb`

Python version: `tf_infer.py`

### TensorFlow optimize

Use TensorRT+TensorFlow to build a new TF graph with optimized TRT-based subgraphs

Jupyter version: `Tensorflow_optimize.ipynb`

Python version: `tf_optimize.py`

## TensorRT part

*Requirements:* It is recommended to run everything inside `tensorrt` docker container (see docker details below)

### TensorRT optimize

Optimize frozen TF graph and prepare a stand-alone TensorRT inference engine

Jupyter version: `TensorRT_optimize.ipynb`

Python version: `trt_optimize.py`

### TensorRT inference

Inference by means of TensorRT

Jupyter version: `TensorRT_infer.ipynb`

Python version: `trt_infer.py`

## Docker

To avoid problems with various versions of the frameworks, it is recommended to use docker containers.

There are two containers with the following Dockerfiles:
 - `tensorflow.Dockerfile` contains TensorFlow 1.10 built against CUDA 10. This container is recommended for all steps from `TensorFlow part`
 - `tensorrt.Dockerfile` contains TensorRT 5.0. This container is recommended for all steps from `TensorRT part`

 You can use either standard docker commands or `docker-compose`. Below is the way using standard commands.

 To build docker containers use `docker_build.sh`

 To run a docker container in bash mode (useful for python scripts) use `docker_run_bash.sh TF` or `docker_run_bash.sh TRT`

 To run a docker container in jupyter mode (useful for jupyer notebooks) use `docker_run_jupyter.sh TF` or `docker_run_jupyter.sh TRT`

 Jupyter notebook password is set in `.env` file

 Jupyter notebook ports:
 - `8881` for TensorFlow container
 - `8882` for TensorRT container
 
 ## Training data

The training is performed on the ImageNet dataset (ILSVRC2012, http://image-net.org). In particular, on the ImageNet subset "Tabby cat" and "Bernese mountain dog" (cats vs dogs).

You can change `TRAIN_DATA_ROOT` and `TRAIN_LIST_FILE` variables according to your localtion of the ImageNet dataset, or create a symlink `/imagenet/` pointing to your location of the ImageNet.
