#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
if [ $1 ]
then
    python main.py --phase train --light True --iteration $1 --save_freq $2; 
else
    python main.py --phase train --light True --iteration 1000 --save_freq 1000;
fi