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
        qid = sent_tokens[3].split()[0]
        if qid in train_topics:
            train_file.write(sent)
        elif qid in dev_topics:
            dev_file.write(sent)
        else:
            test_file.write(sent)

# Create test qrels
doc2sent = {}
with open('data/robust04/robust04_test.csv', mode='r') as robust04_file:
    lines = robust04_file.readlines()
    for line in lines:
        _, _, _, _, _, docid, qid, sentid = line.rstrip().split('\t')
        docid = docid.split('_')[0]
        if qid not in doc2sent.keys():
            doc2sent[qid] = {}
        if docid not in doc2sent[qid].keys():
            doc2sent[qid][docid] = []
        doc2sent[qid][docid].append(sentid)

with open('data/qrels.tfidf.txt', 'w') as tfidf_qrels, \
    open('data/qrels.robust2004.txt', 'r') as original_qrels:
    qrels = original_qrels.readlines()
    for qrel in qrels:
        qid, _, docid, label = qrel.rstrip().split()
        if (qid in test_topics or qid in dev_topics) and docid in doc2sent[qid].keys():
            sentids = doc2sent[qid][docid]
            for sentid in sentids:
                tfidf_qrels.write('{} 0 {} {}\n'.format(qid, sentid, label))
