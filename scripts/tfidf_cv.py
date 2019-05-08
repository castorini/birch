import os
import sys
import json

folds_path = '../Anserini/src/main/resources/fine_tuning'
para_folder = os.path.join(folds_path, 'robust04-paper1-folds.json')

test_fold = int(sys.argv[1])

# Divide topics according to fold parameters
train_topics, dev_topics, test_topics, all_topics = [], [], [], []
with open(para_folder) as f:
    folds = json.load(f)
for i in range(0, len(folds)):
    all_topics.extend(folds[i])
    if i != test_fold:
        train_topics.extend(folds[i])
    else:
        test_topics.extend(folds[i])

dev_topics = train_topics[-25:]
train_topics = train_topics[:100]

assert(len(train_topics) == 100)
assert(len(dev_topics) == 25)
assert(len(test_topics) == 125)
assert(len(all_topics) == 250)

with open('data/tfidf_sents/tfidf_sents.csv', 'r') as sents_file, \
        open('data/tfidf_sents/tfidf_sents_train.csv', 'w') as train_file, \
        open('data/tfidf_sents/tfidf_sents_dev.csv', 'w') as dev_file, \
        open('data/tfidf_sents/tfidf_sents_test.csv', 'w') as test_file:
    sents = sents_file.readlines()
    for sent in sents:
        sent_tokens = sent.rstrip().split('\t')
        # print(sent_tokens)
        qid = sent_tokens[3].split()[0]
        print(qid)
        if qid in train_topics:
            train_file.write(sent)
        elif qid in dev_topics:
            dev_file.write(sent)
        else:
            test_file.write(sent)
