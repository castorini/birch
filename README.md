# Bert-Fine_tune

### Install dependency
```
pip install -r requirements.txt
```

### Train
```
python src/main.py --mode train --batch_size {batch_size} --learning_rate 1e-5 \
                   --num_train_epochs 3 --data_name mb \
                   --eval_steps 1000 --data_path data --learning_rate 3e-5 \
                   --num_train_epochs 3 --pytorch_dump_path saved/saved.{experiment_name} \
                   --output_path out.{experiment} --predict_path predict.{experiment}
```


### Test
```

python src/main.py --mode test --batch_size {batch_size} --learning_rate 1e-5 \
                   --num_train_epochs 3 --data_name robust04 \
                   --eval_steps 1000 --data_path data --learning_rate 3e-5 \
                   --num_train_epochs 3 --load_trained \
                   --pytorch_dump_path saved/saved.{experiment_name} \
                   --output_path out.{experiment} --predict_path predict.{experiment}
```