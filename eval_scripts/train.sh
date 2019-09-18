#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3
no_apex=$4

if [ ! -d "run_logs/${experiment}" ] ; then
    mkdir -p "run_logs/${experiment}"
fi

for i in $(seq 0 4)
    do
        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 0.1 0.1 $i train > "run_logs/${experiment}/eval${i}a.txt"
        if [ $no_apex = "NOAPEX" ]; then
            cat "run_logs/${experiment}/eval${i}a.txt" | tail -n +2 | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}a_best.txt"
        else
            cat "run_logs/${experiment}/eval${i}a.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}a_best.txt"
        fi
        rm "runs/run.${experiment}.cv.train"

        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 0.1 $i train > "run_logs/${experiment}/eval${i}ab.txt"
        if [ $no_apex = "NOAPEX" ]; then
            cat "run_logs/${experiment}/eval${i}ab.txt" | tail -n +2 | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}ab_best.txt"
        else
            cat "run_logs/${experiment}/eval${i}ab.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}ab_best.txt"
        fi
        rm "runs/run.${experiment}.cv.train"

        python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} 3 1.0 1.0 1.0 $i train > "run_logs/${experiment}/eval${i}abc.txt"
        if [ $no_apex = "NOAPEX" ]; then
            cat "run_logs/${experiment}/eval${i}abc.txt" | tail -n +2 | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}abc_best.txt"
        else
            cat "run_logs/${experiment}/eval${i}abc.txt" | sort -k5r,5 -k3,3 | head -1 > "run_logs/${experiment}/${i}abc_best.txt"
        fi
        rm "runs/run.${experiment}.cv.train"
    done
