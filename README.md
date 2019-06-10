 # Birch
 
[ ![Docker Build Status](https://img.shields.io/docker/cloud/build/osirrc2019/birch.svg)](https://hub.docker.com/r/osirrc2019/birch)
[ ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3241945.svg)](https://doi.org/10.5281/zenodo.3241945)
 
 Document ranking via sentence modeling using BERT

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
cd apex && pip install -v --no-cache-dir . && cd ..

# Set up Anserini
git clone https://github.com/castorini/anserini.git
cd anserini && mvn clean package appassembler:assemble
cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..

# Download data and models
wget https://zenodo.org/record/3241945/files/birch_data.tar.gz
tar -xzvf birch_data.tar.gz
```

## Inference

```
python src/main.py --experiment <qa_2cv, mb_2cv, qa_5cv, mb_5cv> --data_path data --collection <robust04_2cv, robust04_5cv> --inference --model_path <models/saved.mb_3, models/saved.qa_2> --load_trained --batch_size <batch_size>
```

Note that this step takes a long time. 
If you don't want to evaluate the pretrained models, you may skip to the next step and evaluate with our predictions under `data/predictions`.

## Evaluation

### BM25+RM3:

```
./eval_scripts/baseline.sh <path/to/anserini> <path/to/index> <2, 5>
```

### Sentence Evidence:

- Compute document score

Set the last argument to True if you want to tune the hyperparameters first.
To use the default hyperparameters, set to False.

```
./eval_scripts/test.sh <qa_2cv, mb_2cv, qa_5cv, mb_5cv> <2, 5> <path/to/anserini> <True, False>
```

- Evaluate with trec_eval

```
./eval_scripts/eval.sh <bm25+rm3_2cv, qa_2cv, mb_2cv, bm25+rm3_5cv, qa_5cv, mb_5cv> <path/to/anserini> qrels.robust2004.txt
```


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
 
 See this [paper](https://dl.acm.org/citation.cfm?id=3308781) for the exact fold settings.
 
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
