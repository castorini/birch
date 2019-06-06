#!/usr/bin/env bash

experiment=$1
num_folds=$2
anserini_path=$3
tune_params=$4

if [ ${num_folds} == '5' ] ; then
    folds_file="robust04-paper2-folds.json"
else
    folds_file="robust04-paper1-folds.json"
fi

if [ ${tune_params} == "True" ] ; then
    declare -a sents=("a" "ab" "abc")

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

            python src/main.py --experiment ${experiment} --folds_file ${folds_file} 3 ${alpha} ${beta} ${gamma} ${j} test
        done
        cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.${i}
    done
else
    if [ ${num_folds} == '5' ] ; then
        ./eval_scripts/${experiment}_eval_5.sh ${experiment} ${anserini_path} ${folds_file}
    else
        # TODO
        ./eval_scripts/${experiment}_eval_2.sh ${experiment} ${anserini_path} ${folds_file}
    fi
fi
