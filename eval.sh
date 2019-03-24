#!/bin/bash
rm -rf cv.MB.folder2.results
for t in `seq 0 1 1`
do
    python3 eval_bert.py 3 1 1 1 $t train >> cv.MB.folder2.results
           
done

