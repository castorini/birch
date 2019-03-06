#!/bin/csh -f
for i in `seq 0.1 0.1 1`;
# for i in 0.75 ;
do
	python eval_bert.py 2 $i 0.5 > "run.robust04.bert.tweet.top$i.txt"
	judgement=qrels.robust2004.txt
	run="run.robust04.bert.tweet.top$i.txt"
	./trec_eval.9.0/trec_eval -q -c -M 1000 ${judgement} ${run} > ${run}.nist.treceval
	map=`tail -29 ${run}.nist.treceval | grep ^map`
	echo $map "$i"
done