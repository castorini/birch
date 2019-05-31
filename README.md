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
pip3 install -r requirements.txt

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir .
```

## Inference

```
# Download models
curl -o saved.tweet_3 "https://www.googleapis.com/storage/v1/b/birch-models/o/saved.tweet_3?alt=media"
curl -o saved.tweet_3 "https://www.googleapis.com/storage/v1/b/birch-models/o/saved.base_mb_2?alt=media"

python src/main.py --experiment <experiment_name> --run_inference --model_path <model_path> --load_trained --predict_path <predict_path> --output_path <output_path>
```

## Evaluation

- Tune hyperparameters

```
./train.sh mb 5
```

- Calculate document score

Set the last argument to True if you want to use your hyperparameters.
To use the default, set to False.

```
./test.sh mb 5 True
```

- Evaluate with trec_eval

```./eval.sh mb ../Anserini/src/main/resources/topics-and-qrels/qrels.robust2004.txt```

### Significance Tests

```
cd eval_scripts
./compare_runs.sh
```

- Runs for all experiments by default
- Modify arrays `experiments`, `collections` and `metrics` if necessary
- Check results for each experiment under `eval_scripts/sig_tests`

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