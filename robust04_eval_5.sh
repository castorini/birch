#!/usr/bin/env bash

experiment=$1
data_path=$2
folds_path=$3
qrels=$4

# MB 5 folders alpha
if [ ! -f "run.mb.cv.a" ] ; then
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0 0 0 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0 0 1 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.7 0 0 2 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.5 0 0 3 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0 0 4 test
     cat run.mb.cv.test.* > run.mb.cv.a
fi

# MB 5 folders alpha + beta
if [ ! -f "run.mb.cv.b" ] ; then
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0 0 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0 1 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0 2 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.5 0.0 0 3 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.2 0 4 test
     cat run.mb.cv.test.* > run.mb.cv.b
fi

# MB 5 folders alpha + beta + gamma
if [ ! -f "run.mb.cv.c" ] ; then
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0.0 0 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0.1 1 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0.1 2 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.5 0.0 0.0 3 test
     python3 eval_bert.py --experiment ${experiment} --data_path ${data_path} --folds_path ${folds_path} --qrels ${qrels} 3 0.6 0.1 0.1 4 test
     cat run.mb.cv.test.* > run.mb.cv.c
fi