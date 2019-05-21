#!/usr/bin/env bash

## MSMARCO-Only-Core17 5 folders alpha
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 0 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 1 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 2 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 3 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 4 test
# cat run.large_msmarco_core17.cv.test.* > run.large_msmarco_core17.cv.a

## MSMARCO-Only-Core17 5 folders alpha + beta
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 0 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 1 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 2 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 3 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 4 test
# cat run.large_msmarco_core17.cv.test.* > run.large_msmarco_core17.cv.b

## MSMARCO-Only-Core17 5 folders alpha + beta + gamma
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0.2 0 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0.8 1 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0.5 0.7 2 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.7 0.4 3 test
# python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0.4 0.4 4 test
# cat run.large_msmarco_core17.cv.test.* > run.large_msmarco_core17.cv.c

 python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 0 all
 mv run.large_msmarco_core17.cv.all run.large_msmarco_core17.cv.a
 python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0 0 all
 mv run.large_msmarco_core17.cv.all run.large_msmarco_core17.cv.b
 python eval_bert.py --experiment large_msmarco_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.6 0.9 0.8 0 all
 mv run.large_msmarco_core17.cv.all run.large_msmarco_core17.cv.c