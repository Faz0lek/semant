#!/bin/bash

# Author: Martin Kostelník
# Brief: Sort input file using language model.
# Date: 03.07.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/semant/sorting
FILE_PATH=$BASE/semant/sorting/example_shuf.txt
MODEL_PATH=$BASE/models/checkpoint516_400000.pth
TOKENIZER_PATH=$BASE/tokenizers/czert-tokenizer
SAVE_PATH=$BASE/semant/sorting/example-result.txt

python -u $SCRIPTS_DIR/sort.py \
    --model-path $MODEL_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --save-path $SAVE_PATH
