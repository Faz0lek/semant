#!/bin/bash

# Author: Martin Kostelník
# Brief: Create test dataset from existing dataset and filter the original dataset.
# Date: 30.05.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/semant/nsp
DATA_PATH=$BASE/data/periodicals_dataset.short.txt
TEST_PATH=$BASE/data/test_dataset_periodicals.txt
FILTERED_PATH=$BASE/data/periodicals_dataset_filtered.txt

python -u $SCRIPTS_DIR/create_test_dataset.py \
    --data $DATA_PATH \
    --out $TEST_PATH \
    --out-filtered $FILTERED_PATH \
    --size 40
