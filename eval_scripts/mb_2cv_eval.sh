#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3
folds_file=$4

# MB 2 folders alpha
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 0 test
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 1 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.a

# MB 2 folders alpha + beta
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 0 test
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 1 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.ab

# MB 2 folders alpha + beta + gamma
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0.1 0 test
 python src/main.py --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 1 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.abc

 rm runs/run.${experiment}.cv.test.*