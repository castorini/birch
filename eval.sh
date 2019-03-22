#!/bin/bash
rm -rf cv.QA.folder2.results
for t in `seq 0 1 4`
do
    for i in `seq 0 0.1 1.0`
    do
        for j in `seq 0 0.1 1.0`
        do
            for k in `seq 0 0.1 1.0`
            do
                python eval_bert.py 3 $i $j $k $t train
                judgement="../qrels.robust2004.txt"
                run="run.QA.cv.train"
                ../trec_eval -q -M 1000 ${judgement} ${run} >run.nist.treceval
                map=`tail -29 run.nist.treceval | grep ^map`
                echo $t "$i" "$j" "$k" $map >> cv.QA.folder2.results
            done
        done 
    done
done

