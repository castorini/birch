#!/usr/bin/env bash

# MSMARCO-random 5 folders alpha
 python eval_bert.py 3 0.54 0 0 0 test
 python eval_bert.py 3 0.47 0 0 1 test
 python eval_bert.py 3 0.54 0 0 2 test
 python eval_bert.py 3 0.5 0 0 3 test
 python eval_bert.py 3 0.43 0 0 4 test
 cat run.msmarco_random.cv.test.* > run.msmarco_random.cv.a

# MSMARCO-random 5 folders alpha + beta
 python eval_bert.py 3 0.51 0.39 0 0 test
 python eval_bert.py 3 0.46 0.34 0 1 test
 python eval_bert.py 3 0.58 0.53 0 2 test
 python eval_bert.py 3 0.51 0.48 0 3 test
 python eval_bert.py 3 0.51 0.24 0 4 test
 cat run.msmarco_random.cv.test.* > run.msmarco_random.cv.b

# MSMARCO-random 5 folders alpha + beta + gamma
 python eval_bert.py 3 0.53 0.31 0.34 0 test
 python eval_bert.py 3 0.51 0.15 0.38 1 test
 python eval_bert.py 3 0.54 0.32 0.27 2 test
 python eval_bert.py 3 0.42 0.16 0.16 3 test
 python eval_bert.py 3 0.53 0.24 0.32 4 test
 cat run.msmarco_random.cv.test.* > run.msmarco_random.cv.c