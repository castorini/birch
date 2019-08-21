# Birch
 
[ ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3372764.svg)](https://doi.org/10.5281/zenodo.3372764)
 
Document ranking via sentence modeling using BERT

Note: 
The results in the arXiv paper [Simple Applications of BERT for Ad Hoc Document Retrieval](https://arxiv.org/abs/1903.10972) have been superseded by the results in the EMNLP'19 paper [Cross-Domain Modeling of Sentence-Level Evidence
for Document Retrieval].
To reproduce the results in the arXiv paper, please follow the instructions [here](https://github.com/castorini/birch/blob/master/reproduce_arxiv.md) instead.

## Environment & Data

```
# Set up environment
pip install virtualenv
virtualenv -p python3.5 birch_env
source birch_env/bin/activate

# Install dependencies
pip install Cython  # jnius dependency
pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Set up Anserini (last reproduced with commit id: f690b5b769d7b0a623e034b31438df126d81b791)
git clone https://github.com/castorini/anserini.git
cd anserini && mvn clean package appassembler:assemble
cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..

# Download data and models
cd data
wget https://zenodo.org/record/3372764/files/emnlp_bert4ir.tar.gz
tar -xzvf emnlp_bert4ir.tar.gz
cd ..
```

Experiment Names:
- large_mb_robust04, large_mb_core17, large_mb_core18
- large_car_mb_robust04, large_car_mb_core17, large_car_mb_core18
- large_msmarco_mb_robust04, large_msmarco_mb_core17, large_msmarco_mb_core18
- large_car_robust04, large_car_core17, large_car_core18
- large_msmarco_robust04, large_msmarco_core17, large_msmarco_core18

## Training

For BERT(MB):

```
export CUDA_VISIBLE_DEVICES=0; experiment=${experiment}; \
nohup python -u src/main.py --mode training --experiment ${experiment} --collection mb \
--local_model <models/bert-large-uncased.tar.gz> \
--local_tokenizer models/bert-large-uncased-vocab.txt --batch_size 16 \
--data_path data --predict_path data/predictions/predict.${experiment} \
--model_path models/saved.${experiment} --eval_steps 1000 --qrels_file qrels.microblog.txt \
--device cuda --output_path logs/out.${experiment} --qrels_file qrels.microblog.txt > logs/${experiment}.log 2>&1 &
```

For BERT(CAR -> MB) and BERT(MS MARCO -> MB):

```
export CUDA_VISIBLE_DEVICES=0; experiment=${experiment}; \
nohup python -u src/main.py --mode training --experiment ${experiment} --collection mb \
--local_model <models/pytorch_msmarco.tar.gz, models/pytorch_car.tar.gz> \
--local_tokenizer models/bert-large-uncased-vocab.txt --batch_size 16 \
--data_path data --predict_path data/predictions/predict.${experiment} \
--model_path models/saved.${experiment} --eval_steps 1000 --qrels_file qrels.microblog.txt \
--device cuda --output_path logs/out.${experiment} --qrels_file qrels.microblog.txt > logs/${experiment}.log 2>&1 &
```

## Inference

For BERT(MB), BERT(CAR -> MB) and BERT(MS MARCO -> MB):

```
export CUDA_VISIBLE_DEVICES=0; experiment=<experiment_name>; \
nohup python -u src/main.py --mode inference --experiment ${experiment} --collection <robust04, core17, core18> \
--load_trained --model_path <models/saved.large_mb_2, models/saved.car_mb_1, models/saved.msmarco_mb_2> \
--batch_size 4 --data_path data --predict_path data/predictions/predict.${experiment} \
--device cuda --output_path logs/out.${experiment} > logs/${experiment}.log 2>&1 &
```

For BERT(CAR) and BERT(MS MARCO):

```
export CUDA_VISIBLE_DEVICES=0; experiment=<experiment_name; \
nohup python -u src/main.py --mode inference --experiment ${experiment} --collection <robust04, core17, core18> \
--local_model <models/pytorch_msmarco.tar.gz, models/pytorch_car.tar.gz> \
--local_tokenizer models/bert-large-uncased-vocab.txt --batch_size 4 \
--data_path data --predict_path data/predictions/predict.${experiment} \
--device cuda --output_path logs/out.${experiment} > logs/${experiment}.log 2>&1 &
```

Note that this step takes a long time. 
If you don't want to evaluate the pretrained models, you may skip to the next step and evaluate with our predictions under `data/predictions`.

## Evaluation

```
experiment=<experiment_name>
collection=<robust04, core17, core18>
anserini_path=<path/to/anserini/root>
data_path=<path/to/data/root>

# Tune hyperparameters
./eval_scripts/train.sh ${experiment} ${collection} ${anserini_path}

# Run experiment
./eval_scripts/test.sh #{experiment} ${collection} ${anserini_path}

# Evaluate with trec_eval
./eval_scripts/eval.sh #{experiment} ${anserini_path} ${data_path}
```