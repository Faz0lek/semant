#!/bin/bash

# Author: Martin Kostelník
# Brief: Train tokenizer on text datacorpus
# Date: 19.06.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/language_modelling
DATA_PATH=$BASE/data/books/books.txt
SAVE_PATH=$BASE/tokenizer

python -u $SCRIPTS_DIR/train_tokenizer.py \
    --data $DATA_PATH \
    --save-path $SAVE_PATH \
    --vocab-size 20000
