#!/bin/bash
python3 -m pip install paddlepaddle-gpu==1.8.2.post97 -i https://mirror.baidu.com/pypi/simple;
python3 -m pip install paddlex -i https://mirror.baidu.com/pypi/simple;
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
mkdir -p dataset/selfie2anime;
unzip ~/data/data48778/308470_627400_bundle_archive.zip -d dataset/selfie2anime;
mkdir -p results/selfie2anime/model
cp -rf data/data52589/* results/selfie2anime/model/
#python main.py;