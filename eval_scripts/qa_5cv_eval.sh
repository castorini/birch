#!/usr/bin/env bash

experiment=$1
anserini_path=$2
folds_file=$3

# QA 5 folders alpha
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.qa.cv.test.* > runs/run.qa.cv.a

# QA 5 folders alpha + beta
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.9 0.6 0 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.qa.cv.test.* > runs/run.qa.cv.ab

# QA 5 folders alpha + beta + gamma
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.9 0.5 0.1 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.qa.cv.test.* > runs/run.qa.cv.abc

 rm runs/run.qa.cv.test.*