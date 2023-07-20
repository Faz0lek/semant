#!/bin/bash

# Author: Martin Kostelník
# Brief: Test model on Babicka book
# Date: 16.07.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/semant/sorting
FILE_PATH=$BASE/data/babicka/babicka.txt
MODEL_PATH=$BASE/models/lm516.pth
TOKENIZER_PATH=$BASE/tokenizers/czert-tokenizer

python -u $SCRIPTS_DIR/test.py \
    --model-path $MODEL_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --file $FILE_PATH
