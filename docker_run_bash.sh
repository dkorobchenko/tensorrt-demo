#!/bin/bash

if [ "$1" = "TF" ]; then

    nvidia-docker run \
        --rm -it \
        -p 6001:6006 \
        -v `pwd`:/demo/ \
        -v /imagenet/:/imagenet/ \
        --name tensorflow-bash \
        trt-demo-tf:latest \
        /bin/bash

elif [ "$1" = "TRT" ]; then

    nvidia-docker run \
        --rm -it \
        -p 6002:6006 \
        -v `pwd`:/demo/ \
        -v /imagenet/:/imagenet/ \
        --name tensorrt-bash \
        trt-demo-trt:latest \

else

    echo "Usage: ./docker_run_bash.sh <TF|TRT>"

fi
