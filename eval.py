import os
import shlex
import subprocess
import sys
import numpy as np


def evaluate(trec_eval_path, predictions_file, qrels_file):
    cmd = trec_eval_path + " {judgement} {output} -m map -m recip_rank -m P.30".format(
        judgement=qrels_file, output=predictions_file)
    pargs = shlex.split(cmd)
    print("running {}".format(cmd))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    if sys.version_info[0] < 3:
        lines = pout.split(b'\n')
    else:
        lines = pout.split(b'\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    p30 = float(lines[2].strip().split()[-1])
    return map, mrr, p30


def get_acc(prediction_index_list, labels):
    acc = sum(np.array(prediction_index_list) == np.array(labels))
    return acc / (len(labels) + 1e-9)


def get_pre_rec_f1(prediction_index_list, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for p, l in zip(prediction_index_list, labels):
        if p == l:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    eps = 1e-8
    precision = tp * 1.0 / (tp + fp + eps)
    recall = tp * 1.0 / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def get_p1(prediction_score_list, labels, data_path, data_name, split):
    f = open(os.path.join(data_path, "{}/{}_{}.csv".format(data_name, data_name, split)))
    a2score_label = {}
    for line, p, l in zip(f, prediction_score_list, labels):
        label, a, b = line.replace("\n", "").split("\t")
        if a not in a2score_label:
            a2score_label[a] = []
        a2score_label[a].append((p, l))

    acc = 0
    no_true = 0
    for a in a2score_label:
        a2score_label[a] = sorted(a2score_label[a], key=lambda x: x[0], reverse=True)
        if a2score_label[a][0][1] > 0:
            acc += 1
        if sum([tmp[1] for tmp in a2score_label[a]]) == 0:
            no_true += 1

    p1 = acc / (len(a2score_label) - no_true)

    return p1
