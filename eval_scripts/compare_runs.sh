#!/usr/bin/env bash

declare -a experiments=("large_mb" "large_car" "large_msmarco" "large_car_mb" "large_msmarco_mb")
declare -a collections=("robust04" "core17" "core18")
declare -a metrics=("map" "P.20" "ndcg_cut.20")

if [ ! -d "sig_tests" ] ; then
    mkdir sig_tests
fi

for e in "${experiments[@]}"
do
    for c in "${collections[@]}"
    do
        if [ ! -f "../run.${e}_${c}.cv.a" ] || [ ! -f "../run.${e}_${c}.cv.b" ] || [ ! -f "../run.${e}_${c}.cv.c" ] ; then
            ./eval_scripts/${c}/${e}_${c}.sh
            echo "${e}_${c}.sh complete"
        else
            echo "${e}_${c}.sh already ran"
        fi

        for m in "${metrics[@]}"
        do
            if [ ! -f "sig_tests/${e}-${c}-${m}-1S.txt" ] ; then
                python3 compare_runs.py \
                --base ../../Anserini/run.${c}.bm25+rm3.topics.${c}.txt --comp ../run.${e}_${c}.cv.a \
                --qrels ../../Anserini/src/main/resources/topics-and-qrels/qrels.${c}.txt \
                --metric ${m} &> temp.txt
                tail -8 temp.txt > "sig_tests/${e}-${c}-${m}-1S.txt"

                echo  "${e}-${c}-${m}-1S.txt complete"
            else
                echo  "${e}-${c}-${m}-1S.txt already exists"
            fi

            if [ ! -f "sig_tests/${e}-${c}-${m}-2S.txt" ] ; then
                python3 compare_runs.py \
                --base ../../Anserini/run.${c}.bm25+rm3.topics.${c}.txt --comp ../run.${e}_${c}.cv.b \
                --qrels ../../Anserini/src/main/resources/topics-and-qrels/qrels.${c}.txt \
                --metric ${m} &> temp.txt
                tail -8 temp.txt > "sig_tests/${e}-${c}-${m}-2S.txt"

                echo  "${e}-${c}-${m}-2S.txt complete"
            else
                echo  "${e}-${c}-${m}-2S.txt already exists"
            fi

            if [ ! -f "sig_tests/${e}-${c}-${m}-3S.txt" ] ; then
                python3 compare_runs.py \
                --base ../../Anserini/run.${c}.bm25+rm3.topics.${c}.txt --comp ../run.${e}_${c}.cv.c \
                --qrels ../../Anserini/src/main/resources/topics-and-qrels/qrels.${c}.txt \
                --metric ${m} &> temp.txt
                tail -8 temp.txt > "sig_tests/${e}-${c}-${m}-3S.txt"

                echo  "${e}-${c}-${m}-3S.txt complete"
            else
                echo  "${e}-${c}-${m}-3S.txt already exists"
            fi
        done
    done
done