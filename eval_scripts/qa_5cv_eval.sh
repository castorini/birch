#!/usr/bin/env bash

experiment=$1
collection=$2
anserini_path=$3
folds_file=$4

# QA 5 folders alpha
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 2 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.a

# QA 5 folders alpha + beta
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.9 0.6 0 2 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.ab

# QA 5 folders alpha + beta + gamma
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 0 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 1 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.9 0.5 0.1 2 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 3 test
 python src/main.py --mode retrieval --experiment ${experiment} --collection ${collection} --anserini_path ${anserini_path} --folds_file ${folds_file} 3 0.8 0 0 4 test
 cat runs/run.${experiment}.cv.test.* > runs/run.${experiment}.cv.abc

 rm runs/run.${experiment}.cv.test.*