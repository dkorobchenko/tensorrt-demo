# https://ngc.nvidia.com
FROM nvcr.io/nvidia/tensorrt:18.09-py3

LABEL maintainer="Dmitry Korobchenko <dkorobchenko@nvidia.com>"

RUN /opt/tensorrt/python/python_setup.sh

RUN pip --no-cache-dir install \
    numpy==1.14.5 \
    imageio==2.3.0 \
    scikit-image==0.14.0 \
    jupyter==1.0.0 \
    matplotlib==3.0.0

COPY run_jupyter.sh /opt/

WORKDIR /demo
