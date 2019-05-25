#!/usr/bin/env bash

# Large-MB 5 folders alpha
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0 0 0 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0 0 1 test
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0 0 2 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0 0 3 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0 0 4 test
 cat run.large_mb_robust04.cv.test.* > run.large_mb_robust04.cv.a

# Large-MB 5 folders alpha + beta
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0.1 0 0 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 1 test
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0.1 0 2 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 3 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 4 test
 cat run.large_mb_robust04.cv.test.* > run.large_mb_robust04.cv.b

# Large-MB 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0.1 0 0 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 1 test
 python eval_bert.py --experiment large_mb_robust04 3 0.4 0.1 0 2 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 3 test
 python eval_bert.py --experiment large_mb_robust04 3 0.3 0.1 0 4 test
 cat run.large_mb_robust04.cv.test.* > run.large_mb_robust04.cv.c