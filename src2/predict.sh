#!/usr/bin/env bash

experiment='model-name'
python src/main.py --mode test --batch_size 8 --data_path data --data_name robust04 \
    --pytorch_dump_path saved.${experiment} --load_trained \
    --trec_eval_path ../Anserini/eval/trec_eval.9.0.4/trec_eval \
    --eval_steps 1000 --output_path out.${experiment} \
    --predict_path predict.${experiment}