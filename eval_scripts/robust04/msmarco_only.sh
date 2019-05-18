#!/usr/bin/env bash

# MSMARCO 5 folders alpha
 python eval_bert.py 3 0.4 0 0 0 test
 python eval_bert.py 3 0.4 0 0 1 test
 python eval_bert.py 3 0.4 0 0 2 test
 python eval_bert.py 3 0.4 0 0 3 test
 python eval_bert.py 3 0.4 0 0 4 test
 cat run.msmarco_only.cv.test.* > run.msmarco_only.cv.a

# MSMARCO 5 folders alpha + beta
 python eval_bert.py 3 0.5 0.8 0 0 test
 python eval_bert.py 3 0.5 0.8 0 1 test
 python eval_bert.py 3 0.5 0.6 0 2 test
 python eval_bert.py 3 0.4 0.4 0 3 test
 python eval_bert.py 3 0.4 0.5 0 4 test
 cat run.msmarco_only.cv.test.* > run.msmarco_only.cv.b

# MSMARCO 5 folders alpha + beta + gamma
 python eval_bert.py 3 0.4 0 0.4 0 test
 python eval_bert.py 3 0.4 0 0.5 1 test
 python eval_bert.py 3 0.5 0 0.9 2 test
 python eval_bert.py 3 0.4 0.1 0.5 3 test
 python eval_bert.py 3 0.4 0.1 0.5 4 test
 cat run.msmarco_only.cv.test.* > run.msmarco_only.cv.c