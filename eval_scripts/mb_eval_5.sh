#!/usr/bin/env bash

experiment=$1
anserini_path=$2
folds_file=$3

# MB 5 folders alpha
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.7 0 0 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.5 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0 0 4 test
 cat runs/run.mb.cv.test.* > runs/run.mb.cv.a

# MB 5 folders alpha + beta
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.5 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.2 0 4 test
 cat runs/run.mb.cv.test.* > runs/run.mb.cv.ab

# MB 5 folders alpha + beta + gamma
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0 0 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0.1 1 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0.1 2 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.5 0 0 3 test
 python src/main.py --experiment ${experiment} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.6 0.1 0.1 4 test
 cat runs/run.mb.cv.test.* > runs/run.mb.cv.abc

 rm runs/run.mb.cv.test.*