#!/usr/bin/env bash
echo 'export CUDA_HOME=/usr/local/cuda-10.0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-10.0/bin/:$PATH' >> ~/.bashrc

source ~/.bashrc