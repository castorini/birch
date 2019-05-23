#!/usr/bin/env bash

 # MSMARCO-Core18 5 folders alpha
 python eval_bert.py --experiment large_msmarco_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.6 0 0 0 all
 mv run.large_msmarco_mb_core18.cv.all run.large_msmarco_mb_core18.cv.a
 # MSMARCO-Core18 5 folders alpha + beta
 python eval_bert.py --experiment large_msmarco_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.7 0.7 0 0 all
 mv run.large_msmarco_mb_core18.cv.all run.large_msmarco_mb_core18.cv.b
 # MSMARCO-Core18 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_msmarco_mb_core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --collection core18 --qrels qrels.core18.txt --bm25_res core18_bm25_rm3_cv_sent.txt --folds_path core18-5-folds.json 3 0.5 0 0 0 all
 mv run.large_msmarco_mb_core18.cv.all run.large_msmarco_mb_core18.cv.c