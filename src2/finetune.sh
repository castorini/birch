#!/usr/bin/env bash

experiment='model-name'
python src/main.py --mode train --batch_size 16 --learning_rate 1e-5 \
    --num_train_epochs 3 --data_path data --data_name mb \
    --pytorch_dump_path saved.${experiment} \
    --trec_eval_path ../Anserini/eval/trec_eval.9.0.4/trec_eval \
    --eval_steps 1000 --output_path out.${experiment} \
    --predict_path predict.${experiment}