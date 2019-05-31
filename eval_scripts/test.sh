#!/usr/bin/env bash

experiment=$1
num_folds=$2
tune_params=$3

if [ ${num_folds} == '5' ] ; then
    folds_path=$"../Anserini/src/main/resources/fine_tuning/robust04-paper2-folds.json"
else
    folds_path=$"../Anserini/src/main/resources/fine_tuning/robust04-paper1-folds.json"
fi

declare -a sents=("a" "b" "c")

for i in "${sents[@]}"
do
    if [ ${tune_params} == "True" ] ; then
        for j in $(seq 0 $((num_folds - 1)))
        do
            while IFS= read -r line
            do
                alpha=$(echo ${line#?} | cut -d" " -f1)
                beta=$(echo ${line#?} | cut -d" " -f2)
                gamma=$(echo ${line#?} | cut -d" " -f3)
            done < "${j}${i}_best.txt"

            python eval_bert.py --experiment ${experiment} --folds_path ${folds_path} 3 ${alpha%?} ${beta%?} ${gamma%?} ${j} test

        done
    else
        if [ ${num_folds} == '5' ] ; then
            ./robust04_eval_5.sh ${experiment} ../Anserini/src/main/resources ${folds_path} qrels.robust2004.txt
        else
            ./robust04_eval_2.sh ${experiment} ../Anserini/src/main/resources ${folds_path} qrels.robust2004.txt
        fi
    fi
done