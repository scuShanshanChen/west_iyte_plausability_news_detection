#!/usr/bin/env bash

DATASET_DIR_PATH=../datasets/
DATASET_FILE_PATH=conceptnet-assertions-5.7.0.csv

SCRIPT_DIR=`pwd`
echo $SCRIPT_DIR

cd "$DATASET_DIR_PATH"

mkdir -p OpenKE
cd OpenKE

if [ ! -f "$DATASET_FILE_PATH" ]; then
    echo File does not exist
    wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
    gunzip conceptnet-assertions-5.7.0.csv.gz
fi

cd "$SCRIPT_DIR"

python3 ./kg_embedding.py --kg_path "$DATASET_DIR_PATH"OpenKE/"$DATASET_FILE_PATH" --saved_dir "$DATASET_DIR_PATH"OpenKE --seed 42

echo Conversion is completed