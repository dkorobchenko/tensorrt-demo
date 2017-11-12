# TensorRT Demo

The demo shows how to build, train and test a conv-net using TensorFlow and then how to port it to TensorRT for fast inference.

## Tensorflow part

*Requirements:* TensorFlow 1.4 (https://www.tensorflow.org/)

`Tensorflow_train_model.ipynb` -> Build and train the network using Tensorflow

`Tensorflow_freeze_graph.ipynb` -> Prepare the network for the inference procedure (mostly required for the further porting to TensorRT)

`Tensorflow_inference.ipynb` -> Inference by means of TensorFlow

## TensorRT part
*Requirements:* TensorRT 3.0 (https://developer.nvidia.com/tensorrt): TensorRT and uff python libs. Python installation packages could be found in the TensorRT archive.

`TensorRT_build_engine.ipynb` -> Optimize frozen TF graph and prepare inference engine with TensorRT

`TensorRT_inference.ipynb` -> Inference by means of TensorRT
