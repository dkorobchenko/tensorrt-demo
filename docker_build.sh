#!/bin/bash

docker build -t trt-demo-trt -f ./docker/tensorrt.Dockerfile ./docker
docker build -t trt-demo-tf -f ./docker/tensorflow.Dockerfile ./docker
