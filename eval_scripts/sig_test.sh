#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3
data_path=$4
metric=$5

birch_path=$(pwd)
cd ${anserini_path}

python3.6 src/main/python/compare_runs.py --base ${birch_path}/runs/run.${collection}.bm25+rm3.txt \
--comparison ${birch_path}/runs/run.${experiment}.cv.a --metric ${metric} \
--qrels ${birch_path}/${data_path}/qrels/qrels.${collection}.txt > temp_sigtext.txt
tail -6 temp_sigtext.txt > ${birch_path}/${data_path}/sigtest/${experiment}_${metric}_a

python3.6 src/main/python/compare_runs.py --base ${birch_path}/runs/run.${collection}.bm25+rm3.txt \
--comparison ${birch_path}/runs/run.${experiment}.cv.ab --metric ${metric} \
--qrels ${birch_path}/${data_path}/qrels/qrels.${collection}.txt > temp_sigtext.txt
tail -6 temp_sigtext.txt > ${birch_path}/${data_path}/sigtest/${experiment}_${metric}_ab

python3.6 src/main/python/compare_runs.py --base ${birch_path}/runs/run.${collection}.bm25+rm3.txt \
--comparison ${birch_path}/runs/run.${experiment}.cv.abc --metric ${metric} \
--qrels ${birch_path}/${data_path}/qrels/qrels.${collection}.txt > temp_sigtext.txt
tail -6 temp_sigtext.txt > ${birch_path}/${data_path}/sigtest/${experiment}_${metric}_abc