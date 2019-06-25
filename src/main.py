import argparse
import random
import os

import numpy as np
import torch

from util import *
from eval import *
from data import DataGenerator

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def train(args):
    if args.load_trained:
        last_epoch, arch, model, tokenizer, scores = load_checkpoint(args.pytorch_dump_path)
    else:
        # May load local file or download from huggingface
        model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                           base_tokenizer=args.local_tokenizer,
                                                           device=args.device)
        last_epoch = 1

    train_dataset = DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, 'train', args.device)
    validate_dataset = DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, 'dev', args.device)
    test_dataset = DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, 'test', args.device)

    optimizer = init_optimizer(model, args.learning_rate, args.warmup_proportion,
                               args.num_train_epochs, train_dataset.data_size, args.batch_size)

    model.train()
    best_score = 0
    step = 0
    for epoch in range(last_epoch, args.num_train_epochs + 1):
        print('Epoch: {}'.format(epoch))
        tr_loss = 0
        while True:
            batch = train_dataset.load_batch()
            if batch is None:
                break
            tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch[:4]
            loss = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                print('Step: {}'.format(step))
                best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path,
                                         best_score, epoch)

            step += 1

        print('[train] loss: {}'.format(tr_loss))
        best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path, best_score,
                                 epoch)

    scores = test(args, split='test')
    print_scores(scores)


def eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score, epoch):
    scores_dev = test(args, split='dev', model=model, test_dataset=validate_dataset)
    print_scores(scores_dev, mode='dev')
    scores_test = test(args, split='test', model=model, test_dataset=test_dataset)
    print_scores(scores_test,  mode='test')

    if scores_dev[1][0] > best_score:
        best_score = scores_dev[1][0]
        model_path = '{}_{}'.format(model_path, epoch)
        save_checkpoint(epoch, model, tokenizer, scores_dev, model_path)

    return best_score


def test(args, split='test', model=None, test_dataset=None):
    if model is None:
        print('Loading model...')
        # May load local file or download from huggingface
        if args.load_trained:
            epoch, arch, model, tokenizer, scores = load_checkpoint(args.pytorch_dump_path)
        else:
            model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                               base_tokenizer=args.local_tokenizer,
                                                               device=args.device)
        assert test_dataset is None
        print('Loading {} set...'.format(split))
        test_dataset =  DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, split, args.device)

    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    output_file = open(args.output_path, 'w')
    predict_file = open(args.predict_path, 'w')

    line_no = 1
    while True:
        batch = test_dataset.load_batch()
        if batch is None:
            break
        tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor = batch
        predictions = model(tokens_tensor, segments_tensor, mask_tensor)
        scores = predictions.cpu().detach().numpy()
        predicted_index = list(torch.argmax(predictions, dim=-1).cpu().numpy())
        predicted_score = list(predictions[:, 1].cpu().detach().numpy())
        prediction_score_list.extend(predicted_score)
        label_batch = list(label_tensor.cpu().detach().numpy())
        label_new = []
        predicted_index_new = []
        if args.data_name == 'mb':
            qids = qid_tensor.cpu().detach().numpy()
            if docid_tensor is not None:
                docids = docid_tensor.cpu().detach().numpy()
            else:
                docids = list(range(line_no, line_no + len(label_batch)))
            for p, qid, docid, s, label in zip(predicted_index, qids, docids, \
                                               scores, label_batch):
                output_file.write('{}\t{}\n'.format(line_no, p))
                predict_file.write('{} Q0 {} {} {} bert\n'.format(qid, docid, line_no, s[1]))
                line_no += 1
        elif args.data_name == 'robust04':
            qids = qid_tensor.cpu().detach().numpy()
            docids = docid_tensor.cpu().detach().numpy()
            assert len(qids) == len(predicted_index)
            for p, l, s, qid, docid in zip(predicted_index, label_batch, scores, qids, docids):
                output_file.write('{} Q0 {} {} {} bert {}\n'.format(qid, docid, line_no, s[1], l))
                line_no += 1
        else:
            if qid_tensor is None:
                qids = list(range(line_no, line_no + len(label_batch)))
            else:
                qids = qid_tensor.cpu().detach().numpy()
            assert len(qids) == len(predicted_index)
            for qid, p, l in zip(qids, predicted_index, label_batch):
                output_file.write('{},{},{}\n'.format(qid, p, l))

        label_new = label_new if len(label_new) > 0 else label_batch
        predicted_index_new = predicted_index_new if len(predicted_index_new) > 0 else predicted_index
        labels.extend(label_new)
        prediction_index_list += predicted_index_new
        del predictions

    output_file.close()
    predict_file.close()
    torch.cuda.empty_cache()
    model.train()

    map, mrr, p30 = evaluate(args.trec_eval_path, predictions_file=args.predict_path, \
                                  qrels_file=os.path.join(args.data_path,
                                                          args.qrels_file))
    return [['map', 'mrr', 'p30'], [map, mrr, p30]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int, help='Number of evaluation steps, -1 for evaluation per epoch')
    parser.add_argument('--data_path', default='data', help='Path to data root directory')
    parser.add_argument('--data_name', default='robust04', help='[mb, robust04]')
    parser.add_argument('--trec_eval_path', default='eval/trec_eval.9.0.4/trec_eval', help='')
    parser.add_argument('--pytorch_dump_path', default='saved.model', help='Path to PyTorch model to save/load')
    parser.add_argument('--load_trained', action='store_true', default=False, help='Load pretrained model')
    parser.add_argument('--local_model', default=None, help='[None, path to local model file]')
    parser.add_argument('--local_tokenizer', default=None, help='[None, path to local vocab file]')
    parser.add_argument('--output_path', default='out.tmp', help='Path to output log')
    parser.add_argument('--predict_path', default='predict.tmp', help='Path to predictions log')
    parser.add_argument('--qrels_file', default='qrels.microblog.txt', help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        scores = test(args)
        print_scores(scores)
