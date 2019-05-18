#!/usr/bin/env bash

# Base-MB 5 folders alpha
 python eval_bert.py --experiment bert_base 3 0.6 0 0 0 test
 python eval_bert.py --experiment bert_base 3 0.6 0 0 1 test
 python eval_bert.py --experiment bert_base 3 0.7 0 0 2 test
 python eval_bert.py --experiment bert_base 3 0.5 0 0 3 test
 python eval_bert.py --experiment bert_base 3 0.6 0 0 4 test
 cat run.bert_base.cv.test.* > run.bert_base.cv.a

# Base-MB 5 folders alpha + beta
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0 0 test
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0 1 test
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0 2 test
 python eval_bert.py --experiment bert_base 3 0.5 0.0 0 3 test
 python eval_bert.py --experiment bert_base 3 0.6 0.2 0 4 test
 cat run.bert_base.cv.test.* > run.bert_base.cv.b

# Base-MB 5 folders alpha + beta + gamma
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0.0 0 test
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0.1 1 test
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0.1 2 test
 python eval_bert.py --experiment bert_base 3 0.5 0.0 0.0 3 test
 python eval_bert.py --experiment bert_base 3 0.6 0.1 0.1 4 test
 cat run.bert_base.cv.test.* > run.bert_base.cv.c