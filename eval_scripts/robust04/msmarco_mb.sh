#!/usr/bin/env bash

### OLD

# MSMARCO-MB 5 folders alpha
 python eval_bert.py --experiment msmarco 3 0.5 0 0 0 test
 python eval_bert.py --experiment msmarco 3 0.5 0 0 1 test
 python eval_bert.py --experiment msmarco 3 0.5 0 0 2 test
 python eval_bert.py --experiment msmarco 3 0.3 0 0 3 test
 python eval_bert.py --experiment msmarco 3 0.4 0 0 4 test
 cat run.msmarco.cv.test.* > run.msmarco.cv.a

# MSMARCO-MB 5 folders alpha + beta
 python eval_bert.py --experiment msmarco 3 0.5 0.4 0 0 test
 python eval_bert.py --experiment msmarco 3 0.5 0.4 0 1 test
 python eval_bert.py --experiment msmarco 3 0.5 0.4 0 2 test
 python eval_bert.py --experiment msmarco 3 0.5 0.4 0 3 test
 python eval_bert.py --experiment msmarco 3 0.5 0.4 0 4 test
 cat run.msmarco.cv.test.* > run.msmarco.cv.b

# MSMARCO-MB 5 folders alpha + beta + gamma
 python eval_bert.py --experiment msmarco 3 0.5 0.2 0.3 0 test
 python eval_bert.py --experiment msmarco 3 0.5 0.2 0.3 1 test
 python eval_bert.py --experiment msmarco 3 0.5 0.2 0.3 2 test
 python eval_bert.py --experiment msmarco 3 0.5 0.2 0.3 3 test
 python eval_bert.py --experiment msmarco 3 0.5 0.3 0.2 4 test
 cat run.msmarco.cv.test.* > run.msmarco.cv.c