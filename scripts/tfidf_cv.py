import os
import sys
import json

folds_path = '../Anserini/src/main/resources/fine_tuning'
para_folder = os.path.join(folds_path, 'robust04-paper2-folds.json')

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
train_topics = train_topics[:175]

assert(len(train_topics) == 175)
assert(len(dev_topics) == 25)
assert(len(test_topics) == 50)
assert(len(all_topics) == 250)

# Create test qrels
sent2doc = {}
with open('data/robust04/robust04_test.csv', 'r') as robust04_file:
    lines = robust04_file.readlines()
    for line in lines:
        _, _, _, _, _, docid, _, sentid = line.rstrip().split('\t')
        docid = docid.split('_')[0]
        sent2doc[sentid] = docid

labels = {}
with open('data/qrels.robust2004.txt', 'r') as original_qrels:
    qrels = original_qrels.readlines()
    for qrel in qrels:
        qid, _, docid, label = qrel.rstrip().split()
        if qid not in labels.keys():
            labels[qid] = {}
        labels[qid][docid] = label

with open('data/tfidf_sents/tfidf_sents.csv', 'r') as sents_file, \
        open('data/tfidf_sents/tfidf_sents_train.csv', 'w') as train_file, \
        open('data/tfidf_sents/tfidf_sents_dev.csv', 'w') as dev_file, \
        open('data/tfidf_sents/tfidf_sents_test.csv', 'w') as test_file, \
        open('data/qrels.tfidf_all.txt', 'w') as tfidf_qrels:
    sents = sents_file.readlines()
    for sent in sents:
        sent_tokens = sent.rstrip().split('\t')
        qid = sent_tokens[3].split()[0]
        sentid = sent_tokens[3].split()[2]
        if qid in train_topics:
            train_file.write(sent)
        elif qid in dev_topics:
            dev_file.write(sent)
        else:
            test_file.write(sent)
        docid = sent2doc[sentid]
        if qid in labels.keys() and docid in labels[qid].keys():
            tfidf_qrels.write(
                '{} {} {} {}\n'.format(qid, 0, sentid, labels[qid][docid]))
