#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
python main.py --phase train --light True --resume True --start_iteration 31000 --iteration 32000 --save_freq 1000;