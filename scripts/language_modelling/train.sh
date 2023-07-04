#!/bin/bash

# Author: Martin Kostelník
# Brief: Train BERT-like language model
# Date: 25.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/semant/language_modelling
SAVE_DIR=$BASE/models

mkdir -p $SAVE_DIR

python -u $SCRIPTS_DIR/train.py \
    --train $BASE/data/books/books-dataset.tst \
    --test $BASE/data/books/books-dataset.tst \
    --split 0.8 \
    --features 72 \
    --mlm-level 2 \
    --epochs 1000 \
    --batch-size 6 \
    --lr 5e-5 \
    --clip 3.0 \
    --view-step 100 \
    --val-step 1000 \
    --warmup-steps 0 \
    --seq-len 40 \
    --fixed-sep \
    --sep \
    --save-path $SAVE_DIR \
    --tokenizer-path $BASE/tokenizers/books-tokenizer-10000
