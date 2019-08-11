#!/usr/bin/env bash

experiment=$1
collection=$2  # robust04_5cv, robust04_2cv, msmarco_top1000_dev, msmarco_expanded_top1000_dev...
folds_file=$3  # robust04-paper1-folds.json, robust04-paper2-folds.json, msmarco-docdev-queries-test-folds.json, msmarco-test2019-queries-folds...
qrels_file=$4  # msmarco-docdev-qrels.tsv...
num_folds=$5
anserini_path=$6
tune_params=$7

declare -a sents=("a" "ab" "abc")

./eval_scripts/train.sh ${experiment} ${num_folds} ${anserini_path}

for i in "${sents[@]}"
do
    for j in $(seq 0 $((num_folds - 1)))
    do
        while IFS= read -r line
        do
            alpha=$(echo ${line#?} | cut -d" " -f1)
            beta=$(echo ${line#?} | cut -d" " -f2)
            gamma=$(echo ${line#?} | cut -d" " -f3)
        done < "log/${experiment}/${j}${i}_best.txt"

        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --qrels_file ${qrels_file} --folds_file ${folds_file} 3 ${alpha} ${beta} ${gamma} ${j} test
    done
    cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.${i}
done
