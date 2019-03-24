from collections import defaultdict
import numpy as np
import operator
import sys
import json
import os

def load_nist_qrels(qrelsF):
    rel_dict = defaultdict(list) 
    all_dict = defaultdict(list)
    nonrel_dict = defaultdict(list)

    with open(qrelsF) as pF:
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
                # print("{} Q0 {} {} {} BERT\n".format(qid, doc, rank, score))
            rank+=1
    for qid in top_doc_dict:
        assert(len(top_doc_dict[qid]) == topK)
    return top_doc_dict, doc_score_dict, sent_dict, q_dict, doc_label_dict

def load_bert_scores(bertF, q_dict, sent_dict):
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
    return score_dict
    

def calc_q_doc_bert(score_dict, runF, test_set,
        topics, top_doc_dict, sent_dict, q_dict,
        bm25_dict, label_dict, topKSent, alpha, beta, gamma=0):
    run_file = open(runF, "w")
    for q in topics:
        doc_score_dict = {}
        assert(len(top_doc_dict[q]) == 1000)
        for d in top_doc_dict[q]:
            scores = score_dict[q][d]
            scores.sort(reverse=True)

            # print q, d, label_dict[q][d], bm25_dict[q][d], \
                # ' '.join(map(str, scores))
            sum_score = 0
            score_list = []
            # rank = 1.0
            weight_list = [1, beta, gamma]

            for s, w in zip(scores[:topKSent], weight_list[:topKSent]):
                score_list.append(s)
                sum_score += s * w
                # rank += 1
            # doc_dict[d] = w * bm25_dict[(q,d)]
            doc_score_dict[d] = alpha * bm25_dict[q][d]+ (1.0-alpha) * sum_score
            # doc_score_dict[d] = np.mean(scores)
            # doc_score_dict[d] = sum(scores[:4])
        doc_score_dict = sorted(doc_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_score_dict:
            run_file.write("{} Q0 {} {} {} BERT\n".format(q, doc, rank, score))
            rank+=1
    run_file.close()


def main():
    topK = int(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    gamma = float(sys.argv[4])
    test_folder_set = int(sys.argv[5])
    mode = sys.argv[6]

    bert_ft = 'MB'
    train_topics,test_topics, all_topics = [], [], []
    with open('../robust04-paper1-folds.json') as f:
        folds = json.load(f)
    for i in range(0, len(folds)):
        all_topics.extend(folds[i])
        if i != test_folder_set:
            train_topics.extend(folds[i])
        else:
            test_topics.extend(folds[i])

    assert(len(train_topics) == 125)
    assert(len(test_topics) == 125)
    assert(len(all_topics) == 250)

    # robust04_rm3_5cv_sent_fields.txt is 5 folder cv sentences
    # prediction 5 folder predict.MB predict.QA

    # robust04_rm3_2cv_sent_fields.txt is 2 folder cv sentences
    # prediction 2 folder predict.MB.2folder
    rel_dict, nonrel_dict, all_dict = load_nist_qrels('../qrels.robust2004.txt')
    top_doc_dict, doc_bm25_dict, sent_dict, q_dict, doc_label_dict = \
        eval_bm25('../robust04_rm3_2cv_sent_fields.txt')

    score_dict = load_bert_scores('predict.'+bert_ft+'.2folder', q_dict,
       sent_dict)

    if mode == 'train':
        # grid search best parameters
        for alpha in np.arange(0, 1.05, 0.1):
            for beta in np.arange(0, 1.05, 0.1):
                for gamma in np.arange(0, 1.05, 0.1):
                    calc_q_doc_bert(score_dict, 'run.'+bert_ft+'.cv.train',
                        test_folder_set, train_topics, top_doc_dict, 
                        sent_dict, q_dict, doc_bm25_dict, doc_label_dict, 
                        topK, alpha, beta, gamma)
                    qrels = '../qrels.robust2004.txt'
                    base = 'run.'+bert_ft+'.cv.train'
                    os.system(f'../trec_eval -M1000 -m map {qrels} {base}> eval.base')
                    with open('eval.base', 'r') as f:
                        for line in f:
                            metric, qid, score = line.split('\t')
                            map_score = float(score)
                            print(test_folder_set,round(alpha,2), 
                                round(beta,2), round(gamma,2), map_score)
    elif mode == 'test':
        calc_q_doc_bert(score_dict, 
            'run.'+bert_ft+'.cv.test.'+str(test_folder_set),
            test_folder_set,
            test_topics, top_doc_dict, sent_dict, q_dict,
            doc_bm25_dict, doc_label_dict, topK, alpha, beta, gamma)
    else:
        calc_q_doc_bert(score_dict, 'run.'+bert_ft+'.cv.all',
            test_folder_set,  all_topics,
            top_doc_dict, sent_dict,q_dict,doc_bm25_dict, doc_label_dict, topK,
            alpha, beta, gamma)


if __name__ == "__main__":
    main()