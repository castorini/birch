#QA
# python eval_bert.py 2 0.83 0.2 0 > run.QA.cv.0
# python eval_bert.py 2 0.83 0.2 1 > run.QA.cv.1
# python eval_bert.py 2 0.83 0.4 2 > run.QA.cv.2
# python eval_bert.py 2 0.83 0.2 3 > run.QA.cv.3
# python eval_bert.py 2 0.83 0.2 4 > run.QA.cv.4
# cat run.QA.cv.* > run.QA.cv

#MB 5 folders
# python eval_bert.py 2 0.66 0.1 0 test 
# python eval_bert.py 2 0.59 0.1 1 test 
# python eval_bert.py 2 0.67 0.1 2 test 
# python eval_bert.py 2 0.55 0.1 3 test 
# python eval_bert.py 2 0.59 0.1 4 test
# cat run.MB.cv.test.* > run.MB.cv

#MB 2 folders
# python eval_bert.py 2 0.63 0.1 0 test 
# python eval_bert.py 2 0.56 0.1 1 test 
# cat run.MB.cv.test.* > run.MB.foder1.cv

# python3 compare_runs.py --base run.robust04.rm3 --comparison run.QA.cv --qrels
# qrels.robust2004.txt --metric map

# python3 compare_runs.py --base run.robust04.rm3 --comparison run.MB.cv --qrels
# qrels.robust2004.txt --metric map

###

## Base-MB 5 folders alpha
# python eval_bert.py 3 0.6 0 0 0 test
# python eval_bert.py 3 0.6 0 0 1 test
# python eval_bert.py 3 0.7 0 0 2 test
# python eval_bert.py 3 0.5 0 0 3 test
# python eval_bert.py 3 0.6 0 0 4 test
# cat run.tweet2014.cv.test.* > run.tweet2014.cv.a
#
## Base-MB 5 folders alpha + beta
# python eval_bert.py 3 0.6 0.1 0 0 test
# python eval_bert.py 3 0.6 0.1 0 1 test
# python eval_bert.py 3 0.6 0.1 0 2 test
# python eval_bert.py 3 0.5 0.0 0 3 test
# python eval_bert.py 3 0.6 0.2 0 4 test
# cat run.tweet2014.cv.test.* > run.tweet2014.cv.b
#
## Base-MB 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.6 0.1 0.0 0 test
# python eval_bert.py 3 0.6 0.1 0.1 1 test
# python eval_bert.py 3 0.6 0.1 0.1 2 test
# python eval_bert.py 3 0.5 0.0 0.0 3 test
# python eval_bert.py 3 0.6 0.1 0.1 4 test
# cat run.tweet2014.cv.test.* > run.tweet2014.cv.c

###

## Large-MB 5 folders alpha
# python eval_bert.py 3 0.4 0 0 0 test
# python eval_bert.py 3 0.3 0 0 1 test
# python eval_bert.py 3 0.4 0 0 2 test
# python eval_bert.py 3 0.3 0 0 3 test
# python eval_bert.py 3 0.3 0 0 4 test
# cat run.bert-large.cv.test.* > run.bert-large.cv.a
#
## Large-MB 5 folders alpha + beta
# python eval_bert.py 3 0.4 0.1 0 0 test
# python eval_bert.py 3 0.3 0.1 0 1 test
# python eval_bert.py 3 0.4 0.1 0 2 test
# python eval_bert.py 3 0.3 0.1 0 3 test
# python eval_bert.py 3 0.3 0.1 0 4 test
# cat run.bert-large.cv.test.* > run.bert-large.cv.b
#
## Large-MB 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.4 0.1 0 0 test
# python eval_bert.py 3 0.3 0.1 0 1 test
# python eval_bert.py 3 0.4 0.1 0 2 test
# python eval_bert.py 3 0.3 0.1 0 3 test
# python eval_bert.py 3 0.3 0.1 0 4 test
# cat run.bert-large.cv.test.* > run.bert-large.cv.c

###

## MSMARCO-MB 5 folders alpha
# python eval_bert.py 3 0.5 0 0 0 test
# python eval_bert.py 3 0.5 0 0 1 test
# python eval_bert.py 3 0.5 0 0 2 test
# python eval_bert.py 3 0.3 0 0 3 test
# python eval_bert.py 3 0.4 0 0 4 test
# cat run.msmarco.cv.test.* > run.msmarco.cv.a

## MSMARCO-MB 5 folders alpha + beta
# python eval_bert.py 3 0.5 0.4 0 0 test
# python eval_bert.py 3 0.5 0.4 0 1 test
# python eval_bert.py 3 0.5 0.4 0 2 test
# python eval_bert.py 3 0.5 0.4 0 3 test
# python eval_bert.py 3 0.5 0.4 0 4 test
# cat run.msmarco.cv.test.* > run.msmarco.cv.b

## MSMARCO-MB 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.5 0.2 0.3 0 test
# python eval_bert.py 3 0.5 0.2 0.3 1 test
# python eval_bert.py 3 0.5 0.2 0.3 2 test
# python eval_bert.py 3 0.5 0.2 0.3 3 test
# python eval_bert.py 3 0.5 0.3 0.2 4 test
# cat run.msmarco.cv.test.* > run.msmarco.cv.c

###

## CAR-MB 5 folders alpha
# python eval_bert.py 3 0.4 0 0 0 test
# python eval_bert.py 3 0.4 0 0 1 test
# python eval_bert.py 3 0.5 0 0 2 test
# python eval_bert.py 3 0.3 0 0 3 test
# python eval_bert.py 3 0.4 0 0 4 test
# cat run.car.cv.test.* > run.car.cv.a

## CAR-MB 5 folders alpha + beta
# python eval_bert.py 3 0.4 0.2 0 0 test
# python eval_bert.py 3 0.4 0.3 0 1 test
# python eval_bert.py 3 0.5 0.3 0 2 test
# python eval_bert.py 3 0.4 0.3 0 3 test
# python eval_bert.py 3 0.4 0.3 0 4 test
# cat run.car.cv.test.* > run.car.cv.b

## CAR-MB 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.4 0.1 0.1 0 test
# python eval_bert.py 3 0.4 0.2 0.1 1 test
# python eval_bert.py 3 0.5 0.3 0 2 test
# python eval_bert.py 3 0.4 0.3 0.1 3 test
# python eval_bert.py 3 0.4 0.2 0.1 4 test
# cat run.car.cv.test.* > run.car.cv.c