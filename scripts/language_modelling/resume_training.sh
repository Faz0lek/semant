#!/bin/bash

# Author: Martin Kostelník
# Brief: Continue training BERT-like language model
# Date: 20.07.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/semant/language_modelling
MODEL_PATH=$BASE/models/lm72.pth
SAVE_DIR=$BASE/models_tmp

mkdir -p $SAVE_DIR

python -u $SCRIPTS_DIR/resume_training.py \
    --model-path $MODEL_PATH \
    --train $BASE/data/books/books-dataset.tst \
    --test $BASE/data/books/books-dataset.tst \
    --split 0.8 \
    --mlm-level 0 \
    --epochs 2 \
    --steps 1000 \
    --batch-size 6 \
    --lr 5e-5 \
    --clip 3.0 \
    --view-step 100 \
    --val-step 1000 \
    --save-path $SAVE_DIR
