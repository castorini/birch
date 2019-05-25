TODO:

- Make variable names easier to understand
- Instructions for data and path organization

---

## Evaluation

Run
```python eval_bert.py num_sentences alpha beta gamma test_fold train```
for all test_fold = {0, 1, 2, 3, 4} three times (once only for alpha, 
once for alpha and beta, once for all alpha, beta and gamma)

Pick the best set of parameters (highest score in the last column)

Run
```python eval_bert.py num_sentences best_set_of_alpha_beta_gamma test_fold test```
for all test_fold = {0, 1, 2, 3, 4}

TODO: turn into script
```
python eval_bert.py 3 0 0 0 1 train > eval0.txt &
cat eval0.txt | sort -k5,5 -r | head -1
```

---

## Data

- Core17: `python core_cv.py --collection core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --output_path core17_sents.txt`
- Core18: `python core_cv.py --collection core18 --index_path /tuna1/indexes/lucene-index.core18.pos+docvectors+rawdocs --output_path core18_sents.txt`

## Evaluation

### Significance Tests

```
cd eval_scripts
./compare_runs.sh
```

- Runs for all experiments by default
- Modify arrays `experiments`, `collections` and `metrics` if necessary
- Check results for each experiment under `eval_scripts/sig_tests`