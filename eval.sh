#!/bin/bash
rm cv.MB.folder2.results
for t in `seq 0 1 1`
do
    for i in `seq 0.5 0.01 1.0`
    do
        for j in `seq 0.1 0.1 0.5`
        do
            python eval_bert.py 2 $i $j $t train
            cat run.MB.cv.train.* > run.MB.cv.train
            judgement=qrels.robust2004.txt
            run="run.MB.cv.train"
            ./trec_eval.9.0/trec_eval -q -M 1000 ${judgement} ${run} > ${run}.nist.treceval
            map=`tail -29 ${run}.nist.treceval | grep ^map`
            echo $t "$i" "$j" $map >> cv.MB.folder2.results
        done 
    done
done
