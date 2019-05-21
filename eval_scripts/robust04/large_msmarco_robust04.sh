#!/usr/bin/env bash

# MSMARCO 5 folders alpha
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0 0 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0 1 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0 2 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0 3 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0 4 test
 cat run.large_msmarco_robust04.cv.test.* > run.large_msmarco_robust04.cv.a

# MSMARCO 5 folders alpha + beta
 python eval_bert.py --experiment large_msmarco_robust04 3 0.5 0.8 0 0 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.5 0.8 0 1 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.5 0.6 0 2 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0.4 0 3 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0.5 0 4 test
 cat run.large_msmarco_robust04.cv.test.* > run.large_msmarco_robust04.cv.b

# MSMARCO 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0.4 0 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0 0.5 1 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.5 0 0.9 2 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0.1 0.5 3 test
 python eval_bert.py --experiment large_msmarco_robust04 3 0.4 0.1 0.5 4 test
 cat run.large_msmarco_robust04.cv.test.* > run.large_msmarco_robust04.cv.c