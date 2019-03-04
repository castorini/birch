from collections import defaultdict
import numpy as np
import operator
import sys

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

def eval_bm25(bm25F, topK = 100):
    doc_score_dict = defaultdict(dict)
    top_doc_dict = defaultdict(list)
    sent_dict = {}
    q_dict = {}
    with open(bm25F) as bF:
        for line in bF:
            _, score, qid, sid, qno, dno = line.strip().split('\t')
            sent_dict[dno] = sid
            q_dict[qno] = qid
            did = sid.split('_')[0]
            doc_score_dict[qid][did] = float(score)
    for qid in doc_score_dict:
        doc_dict = doc_score_dict[qid]
        doc_dict = sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_dict:
            if rank <= topK:
                top_doc_dict[qid].append(doc)
            # elif rank > topK:
                # print qid, 'Q0', doc, rank, score, 'BM25'
            rank+=1
    for qid in top_doc_dict:
        assert(len(top_doc_dict[qid]) == topK)
    return top_doc_dict, doc_score_dict, sent_dict, q_dict

def load_q_doc_bert(bertF, top_doc_dict, sent_dict, q_dict, bm25_dict, topK, w):
    score_dict = defaultdict(dict)
    with open(bertF) as bF:
        for line in bF:
            q, _, d, _, score, _ = line.strip().split()
            q = q_dict[q]
            sent = sent_dict[d]
            doc = sent.split('_')[0]
            score = float(score)
            if doc not in score_dict[q]:
                score_dict[q][doc] = [score]
            else:
                score_dict[q][doc].append(score)
    for q in top_doc_dict:
        doc_score_dict = {}
        assert(len(top_doc_dict[q]) == 100)
        for d in top_doc_dict[q]:
            scores = score_dict[q][d]
            scores.sort(reverse=True)
            # assert(len(scores) > 5) 
            sum_score = 0
            score_list = []
            rank = 1.0
            for s in scores[:topK]:
                score_list.append(s)
                sum_score += s / rank
                rank += 1
            # doc_dict[d] = w * bm25_dict[(q,d)]
            doc_score_dict[d] = w * bm25_dict[q][d] + (1.0-w) * sum_score
            # for s in scores:
            #     if s > 1:
            #         sum_score += s
            #         score_list.append(s)
            # if len(score_list) == 0:
            #     doc_dict[d] = w * bm25_dict[(q,d)]
            # else:
            #     doc_dict[d] = w * bm25_dict[(q,d)] + (1.0-w) * np.mean(score_list)

        doc_score_dict = sorted(doc_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_score_dict:
            print q, 'Q0', doc, rank, score, 'BM25'
            rank+=1

def main():
    topK = int(sys.argv[1])
    w = float(sys.argv[2])
    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    # doc_dict, q_dict, score_dict = load_q_doc_bm25('robust04_bm25_1000_fields.txt')
    top_doc_dict, doc_bm25_dict, sent_dict, q_dict = eval_bm25('robust04_bm25_1000_fields.txt')

    load_q_doc_bert('prediction.trec.1000', top_doc_dict, sent_dict,q_dict,
        doc_bm25_dict, topK, w)

if __name__ == "__main__":
    main()