#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate demo
nohup python train.py > train_stream.log 2>&1 &