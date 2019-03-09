#!/bin/bash
rm cv.results
for t in `seq 0 1 4`
do
	for i in `seq 0.7 0.01 1`
	do
		for j in `seq 0.1 0.1 0.6`
		do
			python eval_bert.py 2 $i $j $t > "run.robust04.bert.tweet.top$i.txt"
			judgement=qrels.robust2004.txt
			run="run.robust04.bert.tweet.top$i.txt"
			./trec_eval.9.0/trec_eval -q -M 1000 ${judgement} ${run} > ${run}.nist.treceval
			map=`tail -29 ${run}.nist.treceval | grep ^map`
			echo $t "$i" "$j" $map >> cv.results
		done
	done
done
