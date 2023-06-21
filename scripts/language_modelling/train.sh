#!/bin/bash

# Author: Martin Kostelník
# Brief: Train NSP model
# Date: 25.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/language_modelling
SAVE_DIR=$BASE/models

mkdir -p $SAVE_DIR

python -u $SCRIPTS_DIR/train.py \
    --train $BASE/data/books/books-dataset.tst \
    --test $BASE/data/books/books-dataset.tst \
    --split 0.8 \
    --features 264 \
    --nsp \
    --mlm \
    --epochs 1000 \
    --batch-size 6 \
    --lr 5e-5 \
    --clip 3.0 \
    --view-step 100 \
    --val-step 500 \
    --warmup-steps 0 \
    --seq-len 128 \
    --save-path $SAVE_DIR
