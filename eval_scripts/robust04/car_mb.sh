#!/usr/bin/env bash

# CAR-MB 5 folders alpha
 python eval_bert.py --experiment car 3 0.4 0 0 0 test
 python eval_bert.py --experiment car 3 0.4 0 0 1 test
 python eval_bert.py --experiment car 3 0.5 0 0 2 test
 python eval_bert.py --experiment car 3 0.3 0 0 3 test
 python eval_bert.py --experiment car 3 0.4 0 0 4 test
 cat run.car.cv.test.* > run.car.cv.a

# CAR-MB 5 folders alpha + beta
 python eval_bert.py --experiment car 3 0.4 0.2 0 0 test
 python eval_bert.py --experiment car 3 0.4 0.3 0 1 test
 python eval_bert.py --experiment car 3 0.5 0.3 0 2 test
 python eval_bert.py --experiment car 3 0.4 0.3 0 3 test
 python eval_bert.py --experiment car 3 0.4 0.3 0 4 test
 cat run.car.cv.test.* > run.car.cv.b

# CAR-MB 5 folders alpha + beta + gamma
 python eval_bert.py --experiment car 3 0.4 0.1 0.1 0 test
 python eval_bert.py --experiment car 3 0.4 0.2 0.1 1 test
 python eval_bert.py --experiment car 3 0.5 0.3 0 2 test
 python eval_bert.py --experiment car 3 0.4 0.3 0.1 3 test
 python eval_bert.py --experiment car 3 0.4 0.2 0.1 4 test
 cat run.car.cv.test.* > run.car.cv.c