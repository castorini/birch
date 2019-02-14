from collections import defaultdict
import numpy as np
import operator

def load_nist_qrels():
    rel_dict = defaultdict(list) 
    all_dict = defaultdict(list)
    nonrel_dict = defaultdict(list)

    with open('qrels.robust2004.txt') as pF:
        for line in pF:
            topic, _, doc, label = line.split()
            all_dict[topic].append(doc)
            if int(label) > 0:
                rel_dict[topic].append(doc)
            else:
                nonrel_dict[topic].append(doc)
    return rel_dict, nonrel_dict, all_dict

def load_q_doc_bm25(bm25F):
    doc_dict = {}
    q_dict = {}
    with open(bm25F) as bF:
        for line in bF:
            _, score, q, d, qid, did, qno, dno = line.strip().split('\t')
            doc_dict[dno] = did
            q_dict[qno] = qid
    return doc_dict, q_dict

def load_q_doc_bert(bertF, doc_dict, qno, q_dict):
    score_dict = defaultdict(lambda:0)
    topic = q_dict[qno]
    with open(bertF) as bF:
        for line in bF:
            q, _, d, _, score, _ = line.strip().split()
            if q != qno:
                continue
            sent = doc_dict[d]
            doc = sent.split('_')[0]
            score = float(score)
            # if score > score_dict[doc]:
            score_dict[doc] += score
    sorted_doc = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
    rank = 1
    for doc, score in sorted_doc:
        print topic, 'Q0', doc, rank, score, 'Sum'
        rank+=1

def main():
    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    doc_dict, q_dict = load_q_doc_bm25('robust04_bm25_test.csv')
    for q in q_dict:
        load_q_doc_bert('prediction.trec', doc_dict, q, q_dict)

if __name__ == "__main__":
    main()