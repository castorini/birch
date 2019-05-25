#!/usr/bin/env bash

# MSMARCO-new 5 folders alpha
 python eval_bert.py --experiment msmarco_new 3 0.4 0 0 0 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0 0 1 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0 0 2 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0 0 3 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0 0 4 test
 cat run.msmarco_new.cv.test.* > run.msmarco_new.cv.a

# MSMARCO 5 folders alpha + beta
 python eval_bert.py --experiment msmarco_new 3 0.4 0.5 0 0 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.5 0 1 test
 python eval_bert.py --experiment msmarco_new 3 0.5 0.5 0 2 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.5 0 3 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.5 0 4 test
 cat run.msmarco_new.cv.test.* > run.msmarco_new.cv.b

# MSMARCO 5 folders alpha + beta + gamma
 python eval_bert.py --experiment msmarco_new 3 0.4 0.1 0.3 0 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.2 0.2 1 test
 python eval_bert.py --experiment msmarco_new 3 0.5 0.4 0.3 2 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.3 0.2 3 test
 python eval_bert.py --experiment msmarco_new 3 0.4 0.2 0.2 4 test
 cat run.msmarco_new.cv.test.* > run.msmarco_new.cv.c