import os
import torch
import random

class DataGenerator(object):
    def __init__(self, data_path, data_name, split):
        super(DataGenerator, self).__init__()
        self.tweet = 1 if data_name == 'mb' else 0
        self.tfidf = 1 if 'tfidf' in data_name else 0
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
        elif self.tfidf:
            for l in self.f:
                label, a, b, ID = l.rstrip().split('\t');
                return label, a, b, ID, 'url'
            return None, None, None, None, None
        else:
            for l in self.f:
                label, sim, a, b, qid, docid, qidx, didx = \
                    l.replace("\n", "").split("\t")
                return label, sim, a, b, qid, docid, qidx, didx
            return None, None, None, None, None, None, None, None


def load_data(data_path, data_name, batch_size, tokenizer, split="train",
              device="cuda", add_url=False, padding=None, max_query_len=10):
    test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
    data_set = []
    while True:
        dataGenerator = DataGenerator(data_path, data_name, split)
        while True:
            label, a, b, ID, url = dataGenerator.get_instance()
            if label is None:
                break

            # Pad query
            if padding:
                query_padding = ['[PAD]'] * (max_query_len - len(a.split(' ')))
                query_padding = ' '.join(query_padding)
                if padding == 'left':
                    a = query_padding + " " + a
                elif padding == 'right':
                    a = a + " " + query_padding
            a = "[CLS] " + a + " [SEP]"

            if add_url:
                b = b + " " + url
            b = b + " [SEP]"

            a_index = tokenize_index(a, tokenizer)
            b_index = tokenize_index(b, tokenizer)

            # Pad sequence
            if padding:
                doc_padding = [0] * (
                            512 - len(a_index) - len(b_index))
                if padding == 'left':
                    b_index = doc_padding + b_index
                elif padding == 'right':
                    b_index = b_index + doc_padding

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

        yield None

    return None


def load_trec_data(data_path, data_name, batch_size, tokenizer, split="train",
   device="cuda", padding=None, max_query_len=10):
    test_batch, testqid_batch, mask_batch, label_batch, qid_batch, docid_batch = [], [], [], [], [], []
    data_set = []
    while True:
        dataGenerator = DataGenerator(data_path, data_name, split)
        while True:
            label, sim, a, b, qno, docno, qidx, didx = \
                dataGenerator.get_instance()
            if label is None:
                break

            # Pad query
            if padding:
                query_padding = ['[PAD]'] * (
                    max_query_len - len(a.split(' ')))
                query_padding = ' '.join(query_padding)
                if padding == 'left':
                    a = query_padding + " " + a
                elif padding == 'right':
                    a = a + " " + query_padding
            a = "[CLS] " + a + " [SEP]"

            b = b + " [SEP]"
            a_index = tokenize_index(a, tokenizer)
            b_index = tokenize_index(b, tokenizer)

            # Pad sequence
            if padding:
                doc_padding = [0] * (
                        512 - len(a_index) - len(b_index))
                if padding == 'left':
                    b_index = doc_padding + b_index
                elif padding == 'right':
                    b_index = b_index + doc_padding

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

        yield None

    return None


def tokenize_index(text, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    # Convert token to vocabulary indices
    tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens
