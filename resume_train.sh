#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64;
if [ $1 ]
then
    python main.py --phase train --light True --resume True  --iteration $1 --save_freq $2 --start_iteration $3;
else
    python main.py --phase train --light True --resume True --iteration 100000 --save_freq 10000 --start_iteration 45000;
fi