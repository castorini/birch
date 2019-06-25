import os
import random

import numpy as np
import torch

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class DataGenerator(object):
    def __init__(self, data_path, data_name, batch_size, tokenizer, split, device='cuda'):
        super(DataGenerator, self).__init__()
        self.data = []
        self.lengths = []

        self.fa = open(os.path.join(data_path, '{}/{}/a.toks'.format(data_name, split)))
        self.fb = open(os.path.join(data_path, '{}/{}/b.toks'.format(data_name, split)))
        self.fsim = open(os.path.join(data_path, '{}/{}/sim.txt'.format(data_name, split)))
        self.fid = open(os.path.join(data_path, '{}/{}/id.txt'.format(data_name, split)))

        for a, b, sim, ID in zip(self.fa, self.fb, self.fsim, self.fid):
            self.data.append([sim.replace('\n', ''), a.replace('\n', ''),
                              b.replace('\n', ''), \
                              ID.replace('\n', '')])
            self.lengths.append(len(b.replace('\n', '').split()))

        np.random.shuffle(self.data)
        self.data_i = 0
        self.data_size = len(self.data)
        self.data_name = data_name
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.start = True

    def get_instance(self):
        ret = self.data[self.data_i % self.data_size]
        self.data_i += 1
        return ret

    def epoch_end(self):
        return self.data_i % self.data_size == 0

    def tokenize_index(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens

    def tokenize_two(self, a, b):
        tokenized_text_a = ['[CLS]'] + self.tokenizer.tokenize(a) + ['[SEP]']
        tokenized_text_b = self.tokenizer.tokenize(b)
        tokenized_text_b = tokenized_text_b[:510 - len(tokenized_text_a)]
        tokenized_text_b.append('[SEP]')
        segments_ids = [0] * len(tokenized_text_a) + [1] * len(tokenized_text_b)
        tokenized_text = tokenized_text_a + tokenized_text_b
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens, segments_ids

    def load_batch(self):
        test_batch, token_type_ids_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            if self.data_name == 'robust04':
                label, a, b, qid, docid = instance
                qid = int(qid)
                docid = int(docid)
                qid_batch.append(qid)
                docid_batch.append(docid)
            else:
                label, a, b, ID = instance
                ls = ID.split()
                if len(ls) > 1:
                    qid, _, docid, _, _, _ = ls
                    docid = int(docid)
                    docid_batch.append(docid)
                else:
                    qid = ID
                qid_batch.append(int(qid))
            combine_index, segments_ids = self.tokenize_two(a, b)
            test_batch.append(torch.tensor(combine_index))
            token_type_ids_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(combine_index)))
            label_batch.append(int(label))
            if len(test_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch,
                                                                batch_first=True,
                                                                padding_value=0).to(
                    self.device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(
                    token_type_ids_batch, batch_first=True,
                    padding_value=0).to(self.device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch,
                                                              batch_first=True,
                                                              padding_value=0).to(
                    self.device)
                label_tensor = torch.tensor(label_batch, device=self.device)
                qid_tensor = torch.tensor(qid_batch, device=self.device)
                docid_tensor = torch.tensor(docid_batch, device=self.device)
                return (tokens_tensor, segments_tensor, mask_tensor,
                        label_tensor, qid_tensor, docid_tensor)

        return None
