#!/usr/bin/env bash

experiment=$1
collection=$2  # robust04_5cv, robust04_2cv, msmarco_top1000_dev, msmarco_expanded_top1000_dev...
folds_file=$3  # robust04-paper1-folds.json, robust04-paper2-folds.json, msmarco-docdev-queries-test-folds.json, msmarco-test2019-queries-folds...
qrels_file=$4  # msmarco-docdev-qrels.tsv...
num_folds=$5
anserini_path=$6

if [ ! -d "log/${experiment}" ] ; then
    mkdir -p "log/${experiment}"
fi

for i in $(seq 0 $((num_folds - 1)))
do
    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --folds_file ${folds_file} --qrels_file ${qrels_file} --anserini_path ${anserini_path} --data_path data 3 1.0 0.1 0.1 $i train > "log/${experiment}/eval${i}a.txt"
    cat "log/${experiment}/eval${i}a.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}a_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --folds_file ${folds_file} --qrels_file ${qrels_file} --anserini_path ${anserini_path} --data_path data 3 1.0 1.0 0.1 $i train > "log/${experiment}/eval${i}ab.txt"
    cat "log/${experiment}/eval${i}ab.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}ab_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --folds_file ${folds_file} --qrels_file ${qrels_file} --anserini_path ${anserini_path} --data_path data 3 1.0 1.0 1.0 $i train > "log/${experiment}/eval${i}abc.txt"
    cat "log/${experiment}/eval${i}abc.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}abc_best.txt"
    rm "runs/run.${experiment}.cv.train"
done