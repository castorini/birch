import operator
from pprint import pprint

# Create query-doc relevance dict
robust04_dict = {}
with open('data/robust04/robust04_test.csv', mode='r') as robust04_file:
    lines = robust04_file.readlines()
    for line in lines:
        label, _, a, b, _, _, qid, docid = line.rstrip().split('\t')
        if qid not in robust04_dict.keys():
            robust04_dict[qid] = {}
        robust04_dict[qid][docid] = [label, a, b]

# Create query-doc TF-IDF dict
tfidf_dict = {}
with open('result/result.csv', mode='r') as scores_file, \
    open('data/tfidf_sents/tfidf_sents.csv', mode='w') as out_sents:
    scores = scores_file.readlines()
    for line in scores:
        qid, _, docid, _, score, _, _ = line.split()
        if qid not in tfidf_dict.keys():
            tfidf_dict[qid] = {}
        tfidf_dict[qid][docid] = float(score)
    for qid in tfidf_dict.keys():
        tfidf_dict[qid] = sorted(tfidf_dict[qid].items(),
                                 key=operator.itemgetter(1), reverse=True)
        # Retrieve relevant docs
        rel_docs = list(filter(lambda x: robust04_dict[qid][x[0]][0] == '1', tfidf_dict[qid]))
        if len(rel_docs) == 0: continue
        rel_docid = rel_docs[0][0]
        robust04_entry = robust04_dict[qid][rel_docid]
        out_sents.write("{}\t{}\t{}\t{}\tQ0\t{}\t0\t0.0\tlucene4lm\turl\n".format(robust04_entry[0],
                                                    robust04_entry[1],
                                                    robust04_entry[2],
                                                    qid, rel_docid))
        # Retrieve non-relevant docs
        non_rel_docs = list(filter(lambda x: robust04_dict[qid][x[0]][0] == '0',
                          tfidf_dict[qid]))
        for i in range(0, 10):
            non_rel_docid = non_rel_docs[i][0]
            robust04_entry = robust04_dict[qid][non_rel_docid]
            out_sents.write("{}\t{}\t{}\t{}\tQ0\t{}\t0\t0.0\tlucene4lm\turl\n".format(robust04_entry[0],
                                                          robust04_entry[1],
                                                          robust04_entry[2],
                                                          qid, non_rel_docid))
