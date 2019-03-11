from collections import defaultdict
import numpy as np
import operator
import sys
import json

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

def eval_bm25(bm25F, topK = 1000):
    doc_score_dict = defaultdict(dict)
    doc_label_dict = defaultdict(dict)
    top_doc_dict = defaultdict(list)
    sent_dict = {}
    q_dict = {}
    with open(bm25F) as bF:
        for line in bF:
            label, score, qid, sid, qno, dno = line.strip().split('\t')
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
            # elif rank > topK:
                # print qid, 'Q0', doc, rank, score, 'BM25'
            rank+=1
    for qid in top_doc_dict:
        assert(len(top_doc_dict[qid]) == topK)
    return top_doc_dict, doc_score_dict, sent_dict, q_dict, doc_label_dict

def calc_q_doc_bert(bertF, runF, topics, top_doc_dict, sent_dict, q_dict, 
        bm25_dict, label_dict, topKSent,alpha, beta):
    score_dict = defaultdict(dict)
    with open(bertF) as bF:
        for line in bF:
            q, _, d, _, score, _, _ = line.strip().split()
            q = q_dict[q]
            sent = sent_dict[d]
            doc = sent.split('_')[0]
            # print q, sent, doc, score, bm25_dict[q][doc], label_dict[q][doc]
            score = float(score)
            if doc not in score_dict[q]:
                score_dict[q][doc] = [score]
            else:
                score_dict[q][doc].append(score)
    
    run_file = open(runF, "w")
    for q in topics:
        doc_score_dict = {}
        assert(len(top_doc_dict[q]) == 1000)
        for d in top_doc_dict[q]:
            scores = score_dict[q][d]
            scores.sort(reverse=True)

            # while len(scores) < max_score_sents:
            #     scores.append(-5)
            # scores = scores[:max_score_sents]
            # print q, d, label_dict[q][d], bm25_dict[q][d], \
                # ' '.join(map(str, scores))
            sum_score = 0
            score_list = []
            # rank = 1.0
            weight_list = [1, beta]

            for s, w in zip(scores[:topKSent], weight_list[:topKSent]):
                score_list.append(s)
                sum_score += s * w
                # rank += 1
            # doc_dict[d] = w * bm25_dict[(q,d)]
            doc_score_dict[d] = alpha * bm25_dict[q][d]+ (1.0-alpha) * sum_score
            # doc_score_dict[d] = np.mean(scores)
            # doc_score_dict[d] = sum(scores)
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
            # print q, 'Q0', doc, rank, score, 'BM25'
            run_file.write("{} Q0 {} {} {} BERT\n".format(q, doc, rank, score))
            rank+=1
    run_file.close()


def main():
    topK = int(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    test_set = int(sys.argv[4])
    mode = sys.argv[5]

    train_topics,test_topics,all_topics = [], [], []
    with open('robust04-paper2-folds.json') as f:
        folds = json.load(f)
    for i in range(0, len(folds)):
        all_topics.extend(folds[i])
        if i != test_set:
            train_topics.extend(folds[i])
        else:
            test_topics.extend(folds[i])

    assert(len(train_topics) == 200)
    assert(len(test_topics) == 50)


    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    top_doc_dict, doc_bm25_dict, sent_dict, q_dict, doc_label_dict = \
        eval_bm25('robust04_bm25_rm3_cv_sent_fields.txt')

    if mode == 'train':
        calc_q_doc_bert('predict.MB', 'run.MB.cv', train_topics,
            top_doc_dict, sent_dict,q_dict,doc_bm25_dict, doc_label_dict, topK,
            alpha, beta)
    elif mode == 'test':
        calc_q_doc_bert('predict.MB', 'run.MB.cv.'+str(test_set), test_topics,
            top_doc_dict, sent_dict,q_dict,doc_bm25_dict, doc_label_dict, topK,
            alpha, beta)
    else:
        calc_q_doc_bert('predict.MB', 'run.MB.cv', all_topics,
            top_doc_dict, sent_dict,q_dict,doc_bm25_dict, doc_label_dict, topK,
            alpha, beta)


if __name__ == "__main__":
    main()