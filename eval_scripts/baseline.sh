#!/usr/bin/env bash

anserini_path=$1
index_path=$2
num_folds=$3

birch_path=$(pwd)
cd ${anserini_path}

if [ ${num_folds} == '5' ] ; then
    folds_path="src/main/resources/fine_tuning/robust04-paper2-folds.json"
    params_path="src/main/resources/fine_tuning/robust04-paper2-folds-map-params.json"
else
    folds_path="src/main/resources/fine_tuning/robust04-paper1-folds.json"
    params_path="src/main/resources/fine_tuning/robust04-paper1-folds-map-params.json"
fi

python3 src/main/python/fine_tuning/reconstruct_robus04_tuned_run.py --index ${index_path} --folds ${folds_path} --params ${params_path}
rm run.robust04.bm25+rm3.fold*
mkdir --parents ${birch_path}/runs
mv run.robust04.bm25+rm3.txt ${birch_path}/runs/run.bm25+rm3_${num_folds}cv.txt
