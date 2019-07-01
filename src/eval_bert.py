from collections import defaultdict
import operator
import os


def eval_bm25(collection_file, topK=1000):
    doc_score_dict = defaultdict(dict)
    doc_label_dict = defaultdict(dict)
    top_doc_dict = defaultdict(list)
    sent_dict = {}
    q_dict = {}
    with open(collection_file) as bF:
        for line in bF:
            label, score, _, _, qid, sid, qno, dno = line.strip().split('\t')
            sent_dict[dno] = sid
            q_dict[qno] = qid
            did = sid.split('_')[0]
            doc_score_dict[qid][did] = float(score)
            doc_label_dict[qid][did] = int(label)
    for qid in doc_score_dict:
        doc_dict = doc_score_dict[qid]
        doc_dict = sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_dict:
            if rank <= topK:
                top_doc_dict[qid].append(doc)
            rank += 1
    return top_doc_dict, doc_score_dict, sent_dict, q_dict, doc_label_dict


def load_bert_scores(pred_file, query_dict, sent_dict):
    score_dict = defaultdict(dict)
    with open(pred_file) as bF:
        for line in bF:
            q, _, d, _, score, _ = line.strip().split()
            if q in query_dict.keys():
                q = query_dict[q]
                sent = sent_dict[d]
                doc = sent.split('_')[0]
                score = float(score)
                if doc not in score_dict[q]:
                    score_dict[q][doc] = [score]
                else:
                    score_dict[q][doc].append(score)
    return score_dict


def calc_q_doc_bert(score_dict, run_file, topics, top_doc_dict, bm25_dict,
                    topKSent, alpha, beta, gamma):
    run_file = open(os.path.join('runs', run_file), "w")
    for q in topics:
        doc_score_dict = {}
        for d in top_doc_dict[q]:
            # print(d)
            scores = score_dict[q][d]
            scores.sort(reverse=True)
            # print(scores)
            sum_score = 0
            score_list = []
            weight_list = [1, beta, gamma]
            for s, w in zip(scores[:topKSent], weight_list[:topKSent]):
                score_list.append(s)
                sum_score += s * w
            doc_score_dict[d] = alpha * bm25_dict[q][d] + (1.0 - alpha) * sum_score

        doc_score_dict = sorted(doc_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_score_dict:
            run_file.write("{} Q0 {} {} {} BERT\n".format(q, doc, rank, score))
            rank += 1

    run_file.close()
