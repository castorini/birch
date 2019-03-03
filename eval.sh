#!/bin/csh -f
for i in `seq 0 0.05 1.05`;
# for i in 1 ;
do
	python eval_bert.py 2 $i > "run.robust04.bert.tweet.top$i.txt"
	judgement=qrels.robust2004.txt
	run="run.robust04.bert.tweet.top$i.txt"
	./trec_eval.9.0/trec_eval -q -c -M 100 ${judgement} ${run} > ${run}.nist.treceval
	map=`tail -29 ${run}.nist.treceval | grep ^map`
	echo $map "lambda$i"
done