#!/usr/bin/env bash

experiment=$1
num_folds=$2
anserini_path=$3
tune_params=$4

if [ ${num_folds} == '5' ] ; then
    folds_file="robust04-paper2-folds.json"
    collection="robust04_5cv"
else
    folds_file="robust04-paper1-folds.json"
    collection="robust04_2cv"
fi

if [ ${tune_params} = true ] ; then
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

            python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 ${alpha} ${beta} ${gamma} ${j} test
        done
        cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.${i}
    done
else
    ./eval_scripts/${experiment}_eval.sh ${experiment} ${collection} ${anserini_path} ${folds_file}
fi
