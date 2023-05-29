#!/bin/bash

# Author: Martin Kostelník
# Brief: Train NSP model
# Date: 25.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/nsp
SAVE_DIR=$BASE/models

mkdir -p $SAVE_DIR

python -u $SCRIPTS_DIR/train.py \
    --data $BASE/data/periodicals_dataset.short.txt \
    --czert \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-path $SAVE_DIR \
    --view-step 100 \
    --split 0.8 \
    --model-path $SAVE_DIR/checkpoint_005.pth
