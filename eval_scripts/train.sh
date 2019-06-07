#!/usr/bin/env bash

experiment=$1
num_folds=$2

if [ ${num_folds} == '5' ] ; then
    folds_file="robust04-paper2-folds.json"
else
    folds_file="robust04-paper1-folds.json"
fi

if [ ! -d "log/${experiment}" ] ; then
    mkdir -p "log/${experiment}"
fi

for i in $(seq 0 $((num_folds - 1)))
do
    python src/main.py --experiment ${experiment} --folds_file ${folds_file} --anserini_path ../Anserini --data_path data 3 1.0 0.1 0.1 $i train > "log/${experiment}/eval${i}a.txt"
    cat "log/${experiment}/eval${i}a.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}a_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --experiment ${experiment} --folds_file ${folds_file} --anserini_path ../Anserini --data_path data 3 1.0 1.0 0.1 ${i} train > "log/${experiment}/eval${i}ab.txt"
    cat "log/${experiment}/eval${i}ab.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}ab_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --experiment ${experiment} --folds_file ${folds_file} --anserini_path ../Anserini --data_path data 3 1.0 1.0 1.0 $i train > "log/${experiment}/eval${i}abc.txt"
    cat "log/${experiment}/eval${i}abc.txt" | sort -k5r,5 -k3,3 | head -1 > "log/${experiment}/${i}abc_best.txt"
    rm "runs/run.${experiment}.cv.train"
done