#!/bin/bash

# Author: Martin Kostelník
# Brief: Create dataset from preprocessed txt file.
# Date: 29.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/language_modelling
DATA_PATH=$BASE/data/books.txt
SAVE_PATH=$BASE/data/dataset.txt

python -u $SCRIPTS_DIR/create_dataset.py \
    --data $DATA_PATH \
    --out $SAVE_PATH \
    --balanced
