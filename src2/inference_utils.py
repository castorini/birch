import shlex
import subprocess
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction


def load_pretrained_model_tokenizer(base_model=None, base_tokenizer=None, device='cuda'):
    if device == 'cuda':
        assert torch.cuda.is_available()

    # Load pre-trained model (weights)
    if base_model is None:
        # Download from huggingface
        base_model = "bert-base-uncased"
    model = BertForNextSentencePrediction.from_pretrained(base_model)

    if base_tokenizer is None:
        # Download from huggingface
        tokenizer = BertTokenizer.from_pretrained(base_model)
    else:
        # Load local vocab file
        tokenizer = BertTokenizer.from_pretrained(base_tokenizer)
    model.to(device)
    return model, tokenizer


def load_checkpoint(filename, device='cpu'):
    print('Load PyTorch model from {}'.format(filename))
    state = torch.load(filename, map_location='cpu') if device == 'cpu' else torch.load(filename)
    return state['epoch'], state['arch'], state['model'], \
           state['tokenizer'], state['scores']


def print_scores(scores, mode='test'):
    print()
    print('[{}] '.format(mode), end='')
    for sn, score in zip(scores[0], scores[1]):
        print('{}: {}'.format(sn, score), end=' ')
    print()


def evaluate(trec_eval_path, predictions_file, qrels_file):
    cmd = trec_eval_path + ' {judgement} {output} -m map -m recip_rank -m P.30'.format(
        judgement=qrels_file, output=predictions_file)
    pargs = shlex.split(cmd)
    print('Running {}'.format(cmd))
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
