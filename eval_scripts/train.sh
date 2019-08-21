#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3

if [ ! -d "run_logs/${experiment}" ] ; then
    mkdir -p "run_logs/${experiment}"
fi

if [[ "${collection}" == "robust04" ]] ; then
    for i in $(seq 0 4)
    do
        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 0.1 0.1 $i train > "run_logs/${experiment}/eval${i}a.txt"
        cat "run_logs/${experiment}/eval${i}a.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}a_best.txt"
        rm "runs/run.${experiment}.cv.train"

        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 0.1 $i train > "run_logs/${experiment}/eval${i}ab.txt"
        cat "run_logs/${experiment}/eval${i}ab.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}ab_best.txt"
        rm "runs/run.${experiment}.cv.train"

        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 1.0 $i train > "run_logs/${experiment}/eval${i}abc.txt"
        cat "run_logs/${experiment}/eval${i}abc.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}abc_best.txt"
        rm "runs/run.${experiment}.cv.train"
    done
else
    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 0.1 0.1 0 train > "run_logs/${experiment}/evala.txt"
    cat "run_logs/${experiment}/evala.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/a_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 0.1 0 train > "run_logs/${experiment}/evalab.txt"
    cat "run_logs/${experiment}/evalab.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/ab_best.txt"
    rm "runs/run.${experiment}.cv.train"

    python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 1.0 0 train > "run_logs/${experiment}/evalabc.txt"
    cat "run_logs/${experiment}/evalabc.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/abc_best.txt"
    rm "runs/run.${experiment}.cv.train"
fi