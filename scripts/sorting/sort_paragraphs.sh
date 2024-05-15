#!/bin/bash

# Author: Martin Kostelník
# Brief: Sort input file paragraphs using language model.
# Date: 03.08.2023

BASE=/home/martin/semant

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/semant/sorting
FILE_PATH=$BASE/data/periodicals/periodical-sample/3f9b00b0-681f-11dc-9c9a-000d606f5dc6.txt
MODEL_PATH=$BASE/models_ft/lm264.pth
TOKENIZER_PATH=$BASE/tokenizers/czert-tokenizer
SAVE_PATH=$BASE/semant/sorting/example-result.txt

python -u $SCRIPTS_DIR/sort_paragraphs.py \
    --file $FILE_PATH \
    --model-path $MODEL_PATH \
    --tokenizer-path $TOKENIZER_PATH
