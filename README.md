 # BERT4retrieval
 
 The commands below assume that Anserini and Birch are located in the same directory.
 
## Extract Data

- Core17: `python core_cv.py --collection core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --output_path core17_sents.txt`
- Core18: `python core_cv.py --collection core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --output_path core18_sents.txt`

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

## BM25 results
| Top K Sentences | Method | Recall | Number of Docs | MAP of Max Sent |
|:---------------:|:------:|:------:|:--------------:|:---------------:|
|       1000      |   RM3  |  0.63  |      720.4     |      0.1974     |
|       1500      |   RM3  |  0.67  |     1065.4     |      0.1985     |
|       1000      |  BM25  |  0.61  |      716.6     |      0.1862     |
|       1500      |  BM25  |  0.66  |     1057.1     |      0.1895     |

## BERT pretrained with TRECQA+WikiQA. Topic title vs. sentence

score = Lambda * bm25_rm3 + (1.0-Lambda) * (bert_high_sent_1 +
bert_high_sent_2/2)

|   MAP (Top1000) |   MAP (Top100) |   Lambda   |
|:------:|:------:|:----------:|
| 0.1825 | 0.2051 | 0    |
| 0.1921 | 0.2106 | 0.05 |
| 0.2027 | 0.2151 | 0.1  |
| 0.2137 | 0.2194 | 0.15 |
| 0.2246 | 0.2233 | 0.2  |
| 0.2359 | 0.2282 | 0.25 |
| 0.2467 | 0.2321 | 0.3  |
| 0.2561 | 0.2354 | 0.35 |
| 0.2654 | 0.2388 | 0.4  |
| 0.2739 | 0.2419 | 0.45 |
| 0.2819 | 0.2451 | 0.5  |
| 0.2878 | 0.2472 | 0.55 |
| 0.2932 | 0.2497 | 0.6  |
| 0.296  | 0.2503 | 0.65 |
| 0.2999 | 0.2522 | 0.7  |
| 0.3011 | 0.2525 | 0.75 |
| 0.3008 | 0.2518 | 0.8  |
| 0.2999 | 0.2511 | 0.85 |
| 0.2975 | 0.2495 | 0.9  |
| 0.2952 | 0.2484 | 0.95 |
| 0.2903 | 0.2451 | 1    |


## BERT pretrained with Tweets

score = Lambda * bm25_rm3 + (1.0-Lambda) * (bert_high_sent_1 +
bert_high_sent_2/2)

|   MAP (Top1000) |   MAP (Top100) |   Lambda   |
|:------:|:------:|:----------:|
| 0.1724 | 0.2378 | 0    |
| 0.1836 | 0.2413 | 0.05 |
| 0.196  | 0.2445 | 0.1  |
| 0.2083 | 0.2478 | 0.15 |
| 0.2212 | 0.2505 | 0.2  |
| 0.2349 | 0.2525 | 0.25 |
| 0.2477 | 0.2543 | 0.3  |
| 0.2593 | 0.2562 | 0.35 |
| 0.2693 | 0.2579 | 0.4  |
| 0.2792 | 0.2588 | 0.45 |
| 0.2864 | 0.2596 | 0.5  |
| 0.2922 | 0.2595 | 0.55 |
| 0.297  | 0.2594 | 0.6  |
| 0.2995 | 0.2594 | 0.65 |
| 0.3013 | 0.2585 | 0.7  |
| 0.3016 | 0.2576 | 0.75 |
| 0.3009 | 0.2557 | 0.8  |
| 0.299  | 0.2537 | 0.85 |
| 0.297  | 0.2515 | 0.9  |
| 0.2945 | 0.2489 | 0.95 |
| 0.2903 | 0.2451 | 1    |


## BERT pretrained with TRECQA+WikiQA using topic description

|   MAP  |   Lambda   |
|:------:|:----------:|
| 0.2020 |   0  |
| 0.2076 | 0.05 |
| 0.2130 |  0.1 |
| 0.2189 | 0.15 |
| 0.2237 |  0.2 |
| 0.2301 | 0.25 |
| 0.2347 |  0.3 |
| 0.2399 | 0.35 |
| 0.2437 |  0.4 |
| 0.2466 | 0.45 |
| 0.2493 |  0.5 |
| 0.2520 | 0.55 |
| 0.2534 |  0.6 |
| 0.2548 | 0.65 |
| 0.2552 |  0.7 |
| 0.2560 | 0.75 |
| 0.2561 |  0.8 |
| 0.2545 | 0.85 |
| 0.2520 |  0.9 |
| 0.2494 | 0.95 |
| 0.2451 |   1  |

## TODO

* Combine BERT scores from tweet and trec_wiki_qa
* re-rank from top BM25+RM3 100 to top 1000 documents
