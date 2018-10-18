#!/bin/bash

jupyter notebook --generate-config

# Create Jupyter Notebook password
# JN_PASSWD env var will be taken from .env file
jn_passwd_hash=$(python -c "from notebook.auth import passwd; print(passwd('$JN_PASSWD'))")
echo "Setting Jupyter password: "$JN_PASSWD
echo "c.NotebookApp.password='$jn_passwd_hash'" >> /root/.jupyter/jupyter_notebook_config.py

jupyter notebook --ip=0.0.0.0 --allow-root
