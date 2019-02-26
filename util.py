from tqdm import tqdm
import random 
import os
import numpy as np
import subprocess
import shlex
import sys

import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForNextSentencePrediction, BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam


def load_pretrained_model_tokenizer(model_type="BertForSequenceClassification", device="cuda", chinese=False, num_labels=2):
    # Load pre-trained model (weights)
    if chinese:
        base_model = "bert-base-chinese"
    else:
        base_model = "bert-base-uncased"
    if model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        # Load pre-trained model tokenizer (vocabulary)
    elif model_type == "BertForNextSentencePrediction":
        model = BertForNextSentencePrediction.from_pretrained(base_model)
    elif model_type == "BertForTokenClassification":
        model = BertForTokenClassification.from_pretrained(base_model, num_labels=num_labels)
    else:
        print("[Error]: unsupported model type")
        return None, None
    
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model.to(device)
    return model, tokenizer

class DataGenerator(object):
    def __init__(self, data_path, data_name, batch_size, tokenizer, split, device="cuda", data_format="trec", add_url=False, label_map=None):
        super(DataGenerator, self).__init__()
        self.data = []
        self.data_format = data_format
        if data_format == "trec":
            self.fa = open(os.path.join(data_path, "{}/{}/a.toks".format(data_name, split)))
            self.fb = open(os.path.join(data_path, "{}/{}/b.toks".format(data_name, split)))
            self.fsim = open(os.path.join(data_path, "{}/{}/sim.txt".format(data_name, split)))
            self.fid = open(os.path.join(data_path, "{}/{}/id.txt".format(data_name, split)))
            if add_url:
                self.furl = open(os.path.join(data_path, "{}/{}/url.txt".format(data_name, split)))
                for a, b, sim, ID, url in zip(self.fa, self.fb, self.fsim, self.fid, self.furl):
                    self.data.append([sim.replace("\n", ""), a.replace("\n", ""), b.replace("\n", ""), \
                            ID.replace("\n", ""), url.replace("\n", "")])
            else:
                for a, b, sim, ID in zip(self.fa, self.fb, self.fsim, self.fid):
                    self.data.append([sim.replace("\n", ""), a.replace("\n", ""), b.replace("\n", ""), \
                            ID.replace("\n", "")])

        elif data_format == "ontonote":
            self.f = open(os.path.join(data_path, "{}/{}.char.bmes".format(data_name, split)))
            label, token = [], []
            self.label_map = {} if label_map is None else label_map
            for l in self.f:
                ls = l.replace("\n", "").split()
                if len(ls) > 1:
                    if ls[1] not in self.label_map:
                        if split == "test" or split == "dev":
                            print("See new label in {} set: {}".format(split, ls[1]))
                        self.label_map[ls[1]] = len(self.label_map)
                    label.append(self.label_map[ls[1]])
                    token.append(ls[0])
                else:
                    self.data.append([token, label])
                    label, token = [], []
            if len(label) > 0:
               self.data.append([token, label])

            print("label_map: {}".format(self.label_map))
        
        else:
            self.f = open(os.path.join(data_path, "{}/{}_{}.csv".format(data_name, data_name, split)))
            for l in self.f:
                ls = l.replace("\n", "").split("\t")
                if len(ls) == 3:
                    self.data.append(ls)
                else:
                    self.data.append([ls[0], ls[1], " ".join(ls[2:])])
        
        np.random.shuffle(self.data)
        self.i = 0
        self.data_size = len(self.data)
        self.add_url = add_url
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.start = True

    def get_instance(self):
        ret = self.data[self.i % self.data_size]
        self.i += 1
        return ret

    def epoch_end(self):
        return self.i % self.data_size == 0

    def tokenize_index(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text.insert(0, "[CLS]")
        tokenized_text.append("[SEP]")
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens
    
    def load_batch(self):
        if self.data_format == "ontonote":
            return self.load_batch_seqlabeling()
        else:
            return self.load_batch_pairclassification()

    def load_batch_seqlabeling(self):
        test_batch, mask_batch, label_batch, token_type_ids_batch = [], [], [], []
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            token, label = instance
            # This line is important. Otherwise self.data will be modified
            label = label[:]
            assert len(token) == len(label)
            label.insert(0, self.label_map["O"])
            label.append(self.label_map["O"])
            token_index = self.tokenize_index(" ".join(token))
            #print(token, token_index, label, len(token_index), len(label))
            assert len(token_index) == len(label)
            segments_ids = [0] * len(token_index)
            test_batch.append(torch.tensor(token_index))
            token_type_ids_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(token_index)))
            label_batch.append(torch.tensor(label))
            if len(test_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(self.device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True, padding_value=0).to(self.device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(self.device)
                label_tensor = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True, padding_value=self.label_map["O"]).to(self.device)
                # label_tensor = torch.tensor(label_batch, device=self.device)
                assert tokens_tensor.shape == segments_tensor.shape
                assert tokens_tensor.shape == mask_tensor.shape
                assert tokens_tensor.shape == label_tensor.shape
                return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)
 
    def load_batch_pairclassification(self):
        test_batch, token_type_ids_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            if len(instance) == 5:
                label, a, b, ID, url = instance
            elif len(instance) == 4:
                label, a, b, ID = instance
            else:
                label, a, b = instance
            if self.add_url:
                b = b + " " + url
            a_index = self.tokenize_index(a)
            b_index = self.tokenize_index(b)
            combine_index = a_index + b_index
            segments_ids = [0] * len(a_index) + [1] * len(b_index)
            test_batch.append(torch.tensor(combine_index))
            token_type_ids_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(combine_index)))
            label_batch.append(int(label))
            if len(instance) >= 4:
                qid, _, docid, _, _, _ = ID.split()
                qid = int(qid)
                docid = int(docid)
                qid_batch.append(qid)
                docid_batch.append(docid)
            if len(test_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(self.device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True, padding_value=0).to(self.device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(self.device)
                label_tensor = torch.tensor(label_batch, device=self.device)
                test_batch, token_type_ids_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
                if len(instance) >= 4:
                    qid_tensor = torch.tensor(qid_batch, device=self.device)
                    docid_tensor = torch.tensor(docid_batch, device=self.device)
                    return (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor)
                else:
                    return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)
 
        return None 


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
        

