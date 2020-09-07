#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
python main.py --phase eval --light True --iteration 32000;