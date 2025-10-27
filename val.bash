#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate demo
nohup python eval.py > test.log 2>&1 &