import json
import os
import random

import numpy as np
import torch

from eval_bert import eval_bm25, load_bert_scores, calc_q_doc_bert
from model.train import train
from model.test import test
from model.utils import print_scores
from args import get_args
from query import query_sents

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def main():
    args, other = get_args()

    experiment = args.experiment
    anserini_path = args.anserini_path
    datasets_path = os.path.join(args.data_path, 'datasets')

    if not os.path.isdir('log'):
        os.mkdir('log')

    if args.mode == 'training':
        train(args)
    elif args.mode == 'inference':
        scores = test(args)
        print_scores(scores)
    else:
        folds_path = os.path.join(anserini_path, 'src', 'main', 'resources', 'fine_tuning', args.folds_file)
        qrels_path = os.path.join(anserini_path, 'src', 'main', 'resources', 'topics-and-qrels', args.qrels_file)

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

        if args.interactive:
            query_sents(args)
            test(args)  # inference over each sentence

        collection_path = os.path.join(datasets_path,
                                       args.collection + '.csv') if not args.interactive else args.interactive_path
        predictions_path = os.path.join(args.data_path, 'predictions',
                                        'predict.' + experiment) if not args.interactive else os.path.join(
            args.data_path, 'predictions', args.predict_path)

        top_doc_dict, doc_bm25_dict, sent_dict, q_dict, doc_label_dict = eval_bm25(collection_path)
        score_dict = load_bert_scores(predictions_path, q_dict, sent_dict)
        topics = test_topics if not args.interactive else list(q_dict.keys())

        print(topics)

        if not os.path.isdir('runs'):
            os.mkdir('runs')

        if mode == 'train':
            # Grid search for best parameters
            for a in np.arange(0.0, alpha, 0.1):
                for b in np.arange(0.0, beta, 0.1):
                    for g in np.arange(0.0, gamma, 0.1):
                        calc_q_doc_bert(score_dict, 'run.' + experiment + '.cv.train',
                                        train_topics, top_doc_dict, doc_bm25_dict,
                                        topK, a, b, g)
                        base = 'runs/run.' + experiment + '.cv.train'
                        os.system('{}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map {} {}> eval.base'.format(anserini_path, qrels_path, base))
                        with open('eval.base', 'r') as f:
                            for line in f:
                                metric, qid, score = line.split('\t')
                                map_score = float(score)
                                print(test_folder_set, round(a, 2),
                                        round(b, 2), round(g, 2), map_score)

        elif mode == 'test':
            calc_q_doc_bert(score_dict,
                            'run.' + experiment + '.cv.test.' + str(test_folder_set),
                            topics, top_doc_dict, doc_bm25_dict, topK, alpha,
                            beta, gamma)
        else:
            calc_q_doc_bert(score_dict, 'run.' + experiment + '.cv.all', all_topics,
                            top_doc_dict, doc_bm25_dict, topK, alpha, beta, gamma)


if __name__ == "__main__":
    main()
