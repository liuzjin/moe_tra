#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate demo
nohup python train.py > train.log 2>&1 &