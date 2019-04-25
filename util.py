from tqdm import tqdm
import random 
import os
import numpy as np
import subprocess
import shlex
import sys

import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam


def load_pretrained_model_tokenizer(model_type="BertForSequenceClassification",
                                    local_model=None, local_tokenizer=None,
                                    device="cuda", chinese=False):
    # Load pre-trained model (weights)
    if local_model is None:
        # Download from huggingface
        if chinese:
            base_model = "bert-base-chinese"
        else:
            base_model = "bert-base-uncased"
    if model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(base_model)
        # Load pre-trained model tokenizer (vocabulary)
    elif model_type == "BertForNextSentencePrediction":
        model = BertForNextSentencePrediction.from_pretrained(base_model)
    else:
        print("[Error]: unsupported model type")
        return None, None

    if local_tokenizer is None:
        # Download from huggingface
        tokenizer = BertTokenizer.from_pretrained(base_model)
    else:
        # Load local vocab file
        tokenizer = BertTokenizer.from_pretrained(local_tokenizer)
    model.to(device)
    return model, tokenizer

def evaluate(predictions_file, qrels_file):
    cmd = "../Anserini/eval/trec_eval.9.0.4/trec_eval {judgement} {output} -m map -m recip_rank -m P.30".format(
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

class DataGenerator(object):
    def __init__(self, data_path, data_name, split):
        super(DataGenerator, self).__init__()
        self.tweet = 1 if data_name == 'mb' else 0
        if self.tweet:
            self.fa = open(os.path.join(data_path, "{}/{}/a.toks".format(data_name, split)))
            self.fb = open(os.path.join(data_path, "{}/{}/b.toks".format(data_name, split)))
            self.fsim = open(os.path.join(data_path, "{}/{}/sim.txt".format(data_name, split)))
            self.fid = open(os.path.join(data_path, "{}/{}/id.txt".format(data_name, split)))
            self.furl = open(os.path.join(data_path, "{}/{}/url_new.txt".format(data_name, split)))
        else:
            self.f = open(os.path.join(data_path, "{}/{}_{}.csv".format(data_name, data_name, split)))

    def get_instance(self):
        if self.tweet:
            for a, b, sim, ID, url in zip(self.fa, self.fb, self.fsim, self.fid, self.furl):
                return sim.replace("\n", ""), a.replace("\n", ""), b.replace("\n", ""), ID.replace("\n", ""), url.replace("\n", "")
            return None, None, None, None, None
        else:
            for l in self.f:
                label, sim, a, b, qid, docid, qidx, didx = \
                    l.replace("\n", "").split("\t")
                return label, sim, a, b, qid, docid, qidx, didx
            return None, None, None, None, None, None, None, None

def load_data(data_path, data_name, batch_size, tokenizer, split="train", device="cuda", add_url=False):
    test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
    data_set = []
    while True:
        dataGenerator = DataGenerator(data_path, data_name, split)
        while True:
            label, a, b, ID, url = dataGenerator.get_instance()
            if label is None:
                break
            a = "[CLS] " + a + " [SEP]"
            if add_url:
                b = b + " " + url + " [SEP]"
            else:
                b = b + " [SEP]"
            a_index = tokenize_index(a, tokenizer)
            b_index = tokenize_index(b, tokenizer)
            combine_index = a_index + b_index
            segments_ids = [0] * len(a_index) + [1] * len(b_index)
            test_batch.append(torch.tensor(combine_index))
            testqid_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(combine_index)))
            label_batch.append(int(label))
            qid, _, docid, _, _, _ = ID.split()
            qid = int(qid)
            docid = int(docid)
            qid_batch.append(qid)
            docid_batch.append(docid)
            if len(test_batch) >= batch_size:
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(testqid_batch, batch_first=True, padding_value=0).to(device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
                label_tensor = torch.tensor(label_batch, device=device)
                qid_tensor = torch.tensor(qid_batch, device=device)
                docid_tensor = torch.tensor(docid_batch, device=device)
                data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor))
                test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
                yield (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor)
 
        if len(test_batch) != 0:
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
            segments_tensor = torch.nn.utils.rnn.pad_sequence(testqid_batch, batch_first=True, padding_value=0).to(device)
            mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
            label_tensor = torch.tensor(label_batch, device=device)
            qid_tensor = torch.tensor(qid_batch, device=device)
            docid_tensor = torch.tensor(docid_batch, device=device)
            data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor))
            test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docqid_batch = [], [], [], [], [], []
            yield (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor) 

        # if split != "train":
        #    break
        yield None 

    return None
    # return data_set

def load_trec_data(data_path, data_name, batch_size, tokenizer, split="train",
   device="cuda"):
    test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
    data_set = []
    while True:
        dataGenerator = DataGenerator(data_path, data_name, split)
        while True:
            label, sim, a, b, qno, docno, qidx, didx = \
                dataGenerator.get_instance()
            if label is None:
                break
            a = "[CLS] " + a + " [SEP]"
            b = b + " [SEP]"
            a_index = tokenize_index(a, tokenizer)
            b_index = tokenize_index(b, tokenizer)
            combine_index = a_index + b_index
            segments_ids = [0] * len(a_index) + [1] * len(b_index)
            combine_index = combine_index[:512]
            segments_ids = segments_ids[:512]
            test_batch.append(torch.tensor(combine_index))
            testqid_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(combine_index)))
            label_batch.append(int(label))
            # qid, _, docid, _, _, _ = ID.split()
            qid = int(qidx)
            docid = int(didx)
            qid_batch.append(qid)
            docid_batch.append(docid)
            if len(test_batch) >= batch_size:
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(testqid_batch, batch_first=True, padding_value=0).to(device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
                label_tensor = torch.tensor(label_batch, device=device)
                qid_tensor = torch.tensor(qid_batch, device=device)
                docid_tensor = torch.tensor(docid_batch, device=device)
                data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor))
                test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
                yield (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor)
 
        if len(test_batch) != 0:
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(device)
            segments_tensor = torch.nn.utils.rnn.pad_sequence(testqid_batch, batch_first=True, padding_value=0).to(device)
            mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(device)
            label_tensor = torch.tensor(label_batch, device=device)
            qid_tensor = torch.tensor(qid_batch, device=device)
            docid_tensor = torch.tensor(docid_batch, device=device)
            data_set.append((tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor))
            test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docqid_batch = [], [], [], [], [], []
            yield (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor) 
        
        # if split != "train":
        #    break
        yield None 

    return None
    # return data_set

def init_optimizer(model, learning_rate, warmup_proportion, num_train_epochs, data_size, batch_size):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = data_size / batch_size * num_train_epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                    lr=learning_rate,
                    warmup=warmup_proportion,
                    t_total=num_train_steps)
    
    return optimizer

def tokenize_index(text, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    # Convert token to vocabulary indices
    tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens

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
