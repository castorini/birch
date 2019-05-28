#!/usr/bin/env bash

experiment=$1
num_folds=$2

declare -a sents=("a" "b" "c")

if [ ${num_folds} == '5' ] ; then
    folds_path=$"../Anserini/src/main/resources/fine_tuning/robust04-paper2-folds.json"
else
    folds_path=$"../Anserini/src/main/resources/fine_tuning/robust04-paper1-folds.json"
fi

for i in $(seq 0 $((num_folds - 1)))
do
    python eval_bert.py --experiment ${experiment} --folds_path ${folds_path} 3 1.0 0.1 0.1 $i train > "eval${i}a.txt"
    cat "eval${i}a.txt" | sort -k5,5 -r | head -1 > "${i}a_best.txt"

    python eval_bert.py --experiment ${experiment} --folds_path ${folds_path} 3 1.0 1.0 0.1 $i train > "eval${i}b.txt"
    cat "eval${i}b.txt" | sort -k5,5 -r | head -1 > "${i}b_best.txt"

    python eval_bert.py --experiment ${experiment} --folds_path ${folds_path} 3 1.0 1.0 1.0 $i train > "eval${i}c.txt"
    cat "eval${i}c.txt" | sort -k5,5 -r | head -1 > "${i}c_best.txt"
done