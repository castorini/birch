#!/usr/bin/env bash

 # Large-Core18 5 folders alpha
 python eval_bert.py --experiment large_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.5 0 0 0 all
 mv run.large_mb_core18.cv.all run.large_mb_core18.cv.a
 # Large-Core18 5 folders alpha + beta
 python eval_bert.py --experiment large_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.5 0.2 0 1 all
 mv run.large_mb_core18.cv.all run.large_mb_core18.cv.b
 # Large-Core18 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.6 0.6 0.1 1 all
 mv run.large_mb_core18.cv.all run.large_mb_core18.cv.c