#!/usr/bin/env bash

collection=$1
index_path=$2
anserini_path=$3
data_path=$4

${anserini_path}/target/appassembler/bin/SearchCollection -topicreader Trec -index ${index_path} -topics "${data_path}/topics/topics.${collection}.txt" -output "runs/run.${collection}.bm25+rm3.txt" -bm25 -rm3
