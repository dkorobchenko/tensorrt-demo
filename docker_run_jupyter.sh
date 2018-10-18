#!/bin/bash

if [ "$1" = "TF" ]; then

    nvidia-docker run \
        --rm -it \
        --env-file .env \
        -p 8881:8888 \
        -v `pwd`:/demo/ \
        -v /imagenet/:/imagenet/ \
        --name tensorflow-jupyter \
        trt-demo-tf:latest \
        /bin/bash /opt/run_jupyter.sh

elif [ "$1" = "TRT" ]; then

    nvidia-docker run \
        --rm -it \
        --env-file .env \
        -p 8882:8888 \
        -v `pwd`:/demo/ \
        -v /imagenet/:/imagenet/ \
        --name tensorrt-jupyter \
        trt-demo-trt:latest \
        /bin/bash /opt/run_jupyter.sh

else

    echo "Usage: ./docker_run_jupyter.sh <TF|TRT>"

fi
