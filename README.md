# Bert-Fine_tune

### Install dependency
```
pip install -r requirements.txt
```

### Train
```
python main.py --mode train --batch_size 8 --eval_steps 1000 --data_name {data} --data_path ./data --learning_rate 3e-5 --num_train_epochs 3 --pytorch_dump_path saved/saved.model --data_format {data_format} --output_path predict.{data} --model_type {model_type}
```


### Test
```
python main.py --mode test --data_name {data} --data_path ./data --pytorch_dump_path saved/saved.model.1 --data_format {data_format} --output_path predict.{data} --model_type {model_type}
```

### Current supported datasets

1. Sequence Classification
  * Movie reviews from [kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)
2. Sequence Pair Classification
  * Information Retrieval datasets from [Anserini](http://anserini.io)
    - Robust04
    - Microblog
  * Semantic Textual Similarity
    - [STS-B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) 
3. Sequence Token Classification
  * Name Entity Recoginition
    - ResumeNER
    - OntoNote 4.0 Chinese
    - MSRA
