#!/usr/bin/env bash

 # MSMARCO-Core17 5 folders alpha
 python eval_bert.py --experiment large_msmarco_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0 0 0 all
 mv run.large_msmarco_mb_core17.cv.all run.large_msmarco_mb_core17.cv.a
 # MSMARCO-Core17 5 folders alpha + beta
 python eval_bert.py --experiment large_msmarco_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0.9 0 0 all
 mv run.large_msmarco_mb_core17.cv.all run.large_msmarco_mb_core17.cv.b
 # MSMARCO-Core17 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_msmarco_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.4 0.3 0.4 0 all
 mv run.large_msmarco_mb_core17.cv.all run.large_msmarco_mb_core17.cv.c