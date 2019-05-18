#!/usr/bin/env bash

## Large-Core17 5 folders alpha
# python eval_bert.py 3 0.4 0 0 0 test
# python eval_bert.py 3 0.4 0 0 1 test
# python eval_bert.py 3 0.3 0 0 2 test
# python eval_bert.py 3 0.4 0 0 3 test
# python eval_bert.py 3 0.4 0 0 4 test
# cat run.large_core17_test.cv.test.* > run.large_core17_test.cv.a

## Large-Core17 5 folders alpha + beta
# python eval_bert.py 3 0.4 0.2 0 0 test
# python eval_bert.py 3 0.5 0.4 0 1 test
# python eval_bert.py 3 0.3 0.2 0 2 test
# python eval_bert.py 3 0.5 0.7 0 3 test
# python eval_bert.py 3 0.4 0.1 0 4 test
# cat run.large_core17_test.cv.test.* > run.large_core17_test.cv.b

## Large-Core17 5 folders alpha + beta + gamma
# python eval_bert.py 3 0.4 0.2 0 0 test
# python eval_bert.py 3 0.5 0.2 0.1 1 test
# python eval_bert.py 3 0.2 0.1 0.1 2 test
# python eval_bert.py 3 0.4 0.2 0.2 3 test
# python eval_bert.py 3 0.5 0.1 0.1 4 test
# cat run.large_core17_test.cv.test.* > run.large_core17_test.cv.c

 python eval_bert.py --experiment large_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.4 0 0 0 all
 python eval_bert.py --experiment large_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.3 0.2 0 0 all
 python eval_bert.py --experiment large_core17_test --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0.2 0.1 0 all
