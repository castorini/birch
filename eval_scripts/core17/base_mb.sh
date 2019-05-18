#!/usr/bin/env bash

## Base-Core17 5 folders alpha
# python eval_bert.py 3 0.7 0 0 0 test
# python eval_bert.py 3 0.7 0 0 1 test
# python eval_bert.py 3 0.6 0 0 2 test
# python eval_bert.py 3 0.6 0 0 3 test
# python eval_bert.py 3 0.6 0 0 4 test
# cat run.base_core17_test.cv.test.* > run.base_core17_test.cv.a

## Base-Core17 5 folders alpha + beta
# python eval_bert.py 3 0.7 0.1 0 0 test
# python eval_bert.py 3 0.7 0.4 0 1 test
# python eval_bert.py 3 0.6 0.1 0 2 test
# python eval_bert.py 3 0.6 0.1 0 3 test
# python eval_bert.py 3 0.6 0.1 0 4 test
# cat run.base_core17_test.cv.test.* > run.base_core17_test.cv.b

## Base-Core17 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.7 0.1 0.1 0 test
# python eval_bert.py 3 0.7 0.1 0.1 1 test
# python eval_bert.py 3 0.6 0.1 0 2 test
# python eval_bert.py 3 0.6 0.1 0 3 test
# python eval_bert.py 3 0.7 0 0.1 4 test
# cat run.base_core17_test.cv.test.* > run.base_core17_test.cv.c

 python eval_bert.py --experiment base_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.7 0 0 0 all
 python eval_bert.py --experiment base_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.7 0.1 0 0 all
 python eval_bert.py --experiment base_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.7 0.1 0.1 0 all