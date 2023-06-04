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
    --train $BASE/data/books/books-dataset.tst \
    --test $BASE/data/books/books-dataset.tst \
    --czert \
    --epochs 1 \
    --batch-size 4 \
    --lr 5e-5 \
    --clip 3\
    --save-path $SAVE_DIR \
    --view-step 100 \
    --val-step 500 \
    --split 0.8
    # --model-path $SAVE_DIR/checkpoint_005.pth
