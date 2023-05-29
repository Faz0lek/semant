#!/bin/bash

# Author: Martin Kostelník
# Brief: Preprocess raw dataset (jsonl)
# Date: 29.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/nsp
DATA_PATH=$BASE/data/periodicals.jsonl
SAVE_PATH=$BASE/data/periodicals.txt

python -u $SCRIPTS_DIR/preprocess_data.py \
    --data $DATA_PATH \
    --out $SAVE_PATH \
    --lang cs \
    --remove-accents
