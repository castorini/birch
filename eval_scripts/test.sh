#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3

declare -a sents=("a" "ab" "abc")

for i in "${sents[@]}"
do
    for j in $(seq 0 4)
        do
            while IFS= read -r line
            do
                alpha=$(echo ${line#?} | cut -d" " -f1)
                beta=$(echo ${line#?} | cut -d" " -f2)
                gamma=$(echo ${line#?} | cut -d" " -f3)
            done < "run_logs/${experiment}/${j}${i}_best.txt"

            python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 ${alpha} ${beta} ${gamma} $j test
        done
        cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.$i
done

