#!/bin/bash
python3 -m pip install paddlepaddle-gpu==1.8.2.post97 -i https://mirror.baidu.com/pypi/simple
mkdir -p dataset/selfie2anime
unzip ~/data/data48778/308470_627400_bundle_archive.zip -d dataset/selfie2anime