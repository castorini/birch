from tqdm import tqdm
import random 
import os 
import numpy as np
import argparse
import subprocess
import shlex
import sys
import pickle 

import torch

from util import *

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
        model, tokenizer = load_pretrained_model_tokenizer(args.model_type, local_model=args.local_model,
                                                            local_tokenizer=args.local_tokenizer,
                                                           device=args.device)
        last_epoch = 1
    train_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "train", args.device)
    validate_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "dev", args.device)
    test_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "test", args.device)
    optimizer = init_optimizer(model, args.learning_rate, args.warmup_proportion, args.num_train_epochs, args.data_size, args.batch_size)
    
    model.train()
    global_step = 0
    best_score = 0
    for epoch in range(last_epoch, args.num_train_epochs + 1):
        tr_loss = 0
        # random.shuffle(train_dataset)
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(tqdm(train_dataset)):
            if batch is None:
                break
            tokens_tensor, segments_tensor, mask_tensor, label_tensor, _, _ = batch
            if args.model_type == "BertForNextSentencePrediction" or args.model_type == "BertForQuestionAnswering":
                # print(tokens_tensor.shape, segments_tensor.shape, mask_tensor.shape, label_tensor.shape)
                loss = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            else:
                loss, logits = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            
            if args.eval_steps > 0 and step % args.eval_steps == 0:
                best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path, best_score, epoch, step, args.model_type)

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path, best_score, epoch, step, args.model_type)

    scores = test(args, split="test")
    print_scores(scores)

def eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score, epoch, arch):
    scores_dev = test(args, split="dev", model=model, tokenizer=tokenizer, test_dataset=validate_dataset, train=True)
    print_scores(scores_dev, mode="dev")
    scores_test = test(args, split="test", model=model, tokenizer=tokenizer, test_dataset=test_dataset, train=True)
    print_scores(scores_test)
     
    if scores_dev[1][0] > best_score:
        best_score = scores_dev[1][0]
        # Save pytorch-model
        model_path = "{}_{}".format(model_path, epoch)
        print("Save PyTorch model to {}".format(model_path))
        save_checkpoint(epoch, arch, model, tokenizer, scores_dev, model_path)

    return best_score

def print_scores(scores, mode="test"):
    print("")
    print("[{}] ".format(mode), end="")
    for sn, score in zip(scores[0], scores[1]):
        print("{}: {}".format(sn, score), end=" ")
    print("")

def save_checkpoint(epoch, arch, model, tokenizer, scores, filename):
    state = {
        'epoch': epoch,
        'arch': arch,
        'model': model,
        'tokenizer': tokenizer, 
        'scores': scores
    }
    torch.save(state, filename)

def load_checkpoint(filename):
    print("Load PyTorch model from {}".format(filename))
    state = torch.load(filename)
    return state['epoch'], state['arch'], state['model'], state['tokenizer'], state['scores']

def test(args, split="test", model=None, tokenizer=None):
    if model is None:
        epoch, arch, model, tokenizer, scores = load_checkpoint(args.pytorch_dump_path)

    if split == 'train':
        # Load MB data
        test_dataset = load_data(args.data_path, args.data_name,
                                      args.batch_size, tokenizer, split,
                                      args.device)
    else:
        # Load Robust04 data
        test_dataset = load_trec_data(args.data_path, args.data_name,
           args.batch_size, tokenizer, split, args.device)
    
    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    f = open(args.output_path, "w")
    f2 = open(args.output_path2, "w")
    lineno = 1
    for batch in test_dataset:
        if batch is None:
            break
        tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor = batch
        predictions = model(tokens_tensor, segments_tensor, mask_tensor)
        scores = predictions.cpu().detach().numpy()
        predicted_index = list(torch.argmax(predictions, dim=1).cpu().numpy())
        prediction_index_list += predicted_index
        predicted_score = list(predictions[:, 1].cpu().detach().numpy())
        prediction_score_list.extend(predicted_score)
        labels.extend(list(label_tensor.cpu().detach().numpy()))
        qids = qid_tensor.cpu().detach().numpy()
        docids = docid_tensor.cpu().detach().numpy()
        for p, qid, docid, s in zip(predicted_index, qids, docids, scores):
            f.write("{}\t{}\n".format(lineno, p))
            f2.write("{} Q0 {} {} {} bert\n".format(qid, docid, lineno, s[1]))
            lineno += 1
        del predictions
    
    f.close()
    f2.close()
    # acc, pre, rec, f1 = 0, 0, 0, 0
    # acc = get_acc(prediction_index_list, labels)
    # p1 = get_p1(prediction_score_list, labels, args.data_path, args.data_name, split)
    # pre, rec, f1 = get_pre_rec_f1(prediction_index_list, labels)
    map, mrr, p30 = evaluate(predictions_file=args.output_path2, \
            qrels_file="./qrels.microblog.txt")

    torch.cuda.empty_cache()
    model.train()

    return [["map", "mrr", "p30"], [map, mrr, p30]]
    # return [["acc", "precision", "recall", "f1"], [acc, pre, rec, f1]]
    # return [["acc", "p@1", "precision", "recall", "f1"], [acc, p1, pre, rec, f1]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--batch_size', default=4, type=int, help='[1, 8, 16, 32]')
    parser.add_argument('--data_size', default=41579, type=int, help='[tweet2014: 41579]')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='')
    parser.add_argument('--data_path', default='data', help='')
    parser.add_argument('--data_name', default='robust04_bm25', help='annotation or youzan_new or tweet')
    parser.add_argument('--pytorch_dump_path', default='saved.model', help='')
    parser.add_argument('--load_trained', action='store_true', default=False, help='')
    parser.add_argument('--chinese', action='store_true', default=False, help='')
    parser.add_argument('--local_model', default=None, help='[None, path to local model file]')
    parser.add_argument('--local_tokenizer', default=None, help='[None, path to local vocab file]')
    parser.add_argument('--eval_steps', default=-1, type=int, help='evaluation per [eval_steps] steps, -1 for evaluation per epoch')
    parser.add_argument('--model_type', default='BertForNextSentencePrediction', help='')
    parser.add_argument('--output_path', default='prediction.tmp', help='')
    parser.add_argument('--output_path2', default='prediction.trec', help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        scores = test(args)
        print_scores(scores)
