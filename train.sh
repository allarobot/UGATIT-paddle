#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
python main.py --phase train --light True --iteration 1000 --save_freq 1000;