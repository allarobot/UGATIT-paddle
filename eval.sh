#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
if [ $1 ]
then
    python main.py --phase eval --light True --iteration $1;
else
    python main.py --phase eval --light True --iteration 45000;
fi