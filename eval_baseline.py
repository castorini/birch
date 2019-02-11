from collections import defaultdict
import numpy as np
import operator

def load_topK_doc_run(topK, topic, runF):
    with open(runF) as rF:
        topDoc = []
        for line in rF:
            t, _, sent, rank, score, run = line.split()
            if topic == t and int(rank) <= topK:
                doc = sent.split('_')[0]
                topDoc.append(doc)
    return topDoc

def rerank_doc(topic, runF):
    score_dict = defaultdict(list) 
    with open(runF) as rF:
        topDoc = []
        for line in rF:
            t, _, sent, rank, score, run = line.split()
            if topic == t:
                doc = sent.split('_')[0]
                score_dict[doc].append(float(score))
    final_score_dict = defaultdict(float)
    for doc in score_dict:
        # final_score_dict[doc] = np.mean(score_dict[doc])
        # final_score_dict[doc] = sum(score_dict[doc])
        final_score_dict[doc] = max(score_dict[doc])
    sorted_doc = sorted(final_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    rank = 1
    for doc, score in sorted_doc:
        print topic, 'Q0', doc, rank, score, 'Sum'
        rank+=1

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

def calc_recall(rel_dict, topK=1500):
    recall_list = []
    doc_cnt_list = []
    for topic in rel_dict:
        topDoc = load_topK_doc_run(topK, topic,
            'run.robust04.bm25.topics.robust04.301-450.601-700.txt')
        ret = 0
        rel = len(rel_dict[topic])
        for doc in set(topDoc):
            if doc in rel_dict[topic]:
                ret+=1
        # print topic, ret*1.0/rel
        recall_list.append(ret*1.0/rel)
        doc_cnt_list.append(len(set(topDoc)))
    print np.mean(recall_list)
    print np.mean(doc_cnt_list)


def main():
    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    calc_recall(rel_dict)
    return
    for topic in rel_dict:
        rerank_doc(topic,'run.robust04.bm25.topics.robust04.301-450.601-700.txt')

if __name__ == "__main__":
    main()