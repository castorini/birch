 # Birch
 
 Document ranking via sentence modeling using BERT

 ---
<!--## Extract Data

- Core17: `python core_cv.py --collection core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --output_path core17_sents.txt`
- Core18: `python core_cv.py --collection core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --output_path core18_sents.txt` -->

## Environment

```
pip install virtualenv
virtualenv -p python3.5 birch_env
source birch_env/bin/activate
pip install pytorch-pretrained-bert==0.4.0 ???

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir .
cd ..
```

## Inference

```
# Download data
mkdir -p data/datasets
curl -o data/datasets/robust04.csv "https://www.googleapis.com/download/storage/v1/b/birch_data/o/datasets%2Frobust04_test.csv?alt=media"

# Download models
mkdir models
curl -o models/saved.mb_3 "https://www.googleapis.com/download/storage/v1/b/birch_data/o/birch_models%2Fsaved.mb_3?alt=media"
curl -o models/saved.qa_3 "https://www.googleapis.com/download/storage/v1/b/birch_data/o/birch_models%2Fsaved.qa_3?alt=media"

python src/main.py --experiment <experiment_name> --inference --model_path <model_path> --load_trained
```

If you don't want to evaluate the pretrained models, download our predictions here and skip to the next step:

```
# Download predictions
mkdir -p data/predictions
curl -o data/predictions/predict.mb "https://www.googleapis.com/download/storage/v1/b/birch_data/o/birch_predictions%2Fpredict.mb?alt=media"
curl -o data/predictions/predict.qa_5 "https://www.googleapis.com/download/storage/v1/b/birch_data/o/birch_predictions%2Fpredict.qa_5?alt=media"
curl -o data/predictions/predict.qa_2 "https://www.googleapis.com/download/storage/v1/b/birch_data/o/birch_predictions%2Fpredict.qa_2?alt=media"
```

Note that there might be a very slight difference in the predicted scores due to non-determinism, but it is negligible in evaluation.

## Evaluation

- Tune hyperparameters

```
./eval_scripts/train.sh <qa,mb> <5>
```

- Calculate document score

Set the last argument to True if you want to use the hyperparameters learned in the previous step.
To use the default hyperparameters, set to False.

```
./eval_scripts/test.sh <qa,mb> <5> <anserini_path> <True,False>
```

- Evaluate with trec_eval

```./eval_scripts/eval.sh <qa,mb> <anserini_path> qrels.robust2004.txt```

---

## Result on Robust04
 
  - "Paper 1" based on two-fold CV:
 
|        Model        | AP     | P@20   |
|:-------------------:|:------:|:------:|
|  Paper 1 (two fold) | 0.2971 | 0.3948 |
| BM25+RM3 (Anserini) | 0.2987 | 0.2871 |         
|     1S: BERT(QA)    | 0.3014 | 0.3928 |         
|     2S: BERT(QA)    | 0.3003 | 0.3948 |         
|     3S: BERT(QA)    | 0.3003 | 0.3948 |         
|     1S: BERT(MB)    | 0.3241 | 0.4217 |         
|     2S: BERT(MB)    | 0.3240 | 0.4209 |         
|     3S: BERT(MB)    | **0.3244** | **0.4219** |   
 
 - "Paper 2" based on five-fold CV:
 
|        Model        | AP     | P@20   |
|:-------------------:|:------:|:------:|
| Paper 2 (five fold) |  0.272 |  0.386 |
| BM25+RM3 (Anserini) | 0.3033 | 0.3974 |         
|     1S: BERT(QA)    | 0.3102 | 0.4068 |         
|     2S: BERT(QA)    | 0.3090 | 0.4064 |         
|     3S: BERT(QA)    | 0.3090 | 0.4064 |         
|     1S: BERT(MB)    | 0.3266 | 0.4245 |         
|     2S: BERT(MB)    | **0.3278** | 0.4267 |         
|     3S: BERT(MB)    | **0.3278** | **0.4287** |         
 
 See this [paper](https://dl.acm.org/citation.cfm?id=3308781) for exact fold settings.
 
 ---

**How do I cite this work?**

```
@article{yang2019simple,
  title={Simple Applications of BERT for Ad Hoc Document Retrieval},
  author={Yang, Wei and Zhang, Haotian and Lin, Jimmy},
  journal={arXiv preprint arXiv:1903.10972},
  year={2019}
}
```