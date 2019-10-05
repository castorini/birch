# TREC Deep Learning Track

## Document Ranking

### Data

- Move all data into a folder named `trec_dl` under the main data directory


```
python src/msmarco_cv.py --collection msmarco --output_path data/trec_dl/msmarco_test_sents.csv
```


```
python src/main.py --mode retrieval --experiment bm25_marcomb --collection msmarco_top1000_test \
--folds_file msmarco-test2019-queries-folds.json 3 0.4 0.1 0.1 0 all

python src/main.py --mode retrieval --experiment bm25exp_marcomb --collection msmarco_expanded_top1000_test \
--folds_file msmarco-test2019-queries-folds.json 3 0.3 0.1 0.1 0 all

python src/main.py --mode retrieval --experiment bm25exp_marco --collection msmarco_expanded_top1000_test \
--folds_file msmarco-test2019-queries-folds.json 3 0.6 0.3 0.6 0 all
```