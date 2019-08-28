import json
import os
import random

import numpy as np
import torch

from eval_bert import eval_bm25, load_bert_scores, calc_q_doc_bert
from model.train import train
from model.test import test
from args import get_args

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

import warnings
warnings.filterwarnings('ignore')


def main():
    args, other = get_args()

    experiment = args.experiment
    anserini_path = args.anserini_path
    datasets_path = os.path.join(args.data_path, 'datasets')

    if args.mode == 'training':
        train(args)
    elif args.mode == 'inference':
        test(args)
    else:
        if args.interactive:
            # TODO: sync with HiCAL
            from utils.query import query_sents, visualize_scores

            sentid2text, hits = query_sents(args, K=10)
            test(args)

        else:
            folds_path = os.path.join(args.data_path, 'folds', '{}-folds.json'.format(args.collection))
            qrels_path = os.path.join(args.data_path, 'qrels', 'qrels.{}.txt'.format(args.collection))

            topK = int(other[0])
            alpha = float(other[1])
            beta = float(other[2])
            gamma = float(other[3])
            test_folder_set = int(other[4])
            mode = other[5]

            # Divide topics according to fold parameters
            train_topics, test_topics, all_topics = [], [], []
            with open(folds_path) as f:
                folds = json.load(f)
            for i in range(0, len(folds)):
                all_topics.extend(folds[i])
                if i != test_folder_set:
                    train_topics.extend(folds[i])
                else:
                    test_topics.extend(folds[i])

            collection_path = os.path.join(datasets_path, '{}_sents.csv'.format(args.collection))
            predictions_path = os.path.join(args.data_path, 'predictions', 'predict.' + experiment)

            top_doc_dict, doc_bm25_dict, sent_dict, q_dict, doc_label_dict = eval_bm25(collection_path)
            score_dict = load_bert_scores(predictions_path, q_dict, sent_dict)

            if not os.path.isdir('runs'):
                os.mkdir('runs')

            if mode == 'train':
                if args.collection == 'robust04':
                    topics = train_topics if not args.interactive else list(q_dict.keys())
                else:
                    topics = all_topics
                # Grid search for best parameters
                for a in np.arange(0.0, alpha, 0.1):
                    for b in np.arange(0.0, beta, 0.1):
                        for g in np.arange(0.0, gamma, 0.1):
                            calc_q_doc_bert(score_dict, 'run.' + experiment + '.cv.train', topics, top_doc_dict, doc_bm25_dict, topK, a, b, g)
                            base = 'runs/run.' + experiment + '.cv.train'
                            os.system('{}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map {} {}> eval.base'.format(anserini_path, qrels_path, base))
                            with open('eval.base', 'r') as f:
                                for line in f:
                                    metric, qid, score = line.split('\t')
                                    map_score = float(score)
                                    print(test_folder_set, round(a, 2), round(b, 2), round(g, 2), map_score)

            elif mode == 'test':
                if args.collection == 'robust04':
                    topics = test_topics if not args.interactive else list(q_dict.keys())
                else:
                    topics = all_topics
                calc_q_doc_bert(score_dict, 'run.' + experiment + '.cv.test.' + str(test_folder_set), topics, top_doc_dict, doc_bm25_dict, topK, alpha, beta, gamma)


if __name__ == '__main__':
    main()
