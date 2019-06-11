import argparse

from src2.data import load_data, load_trec_data
from src2.eval import evaluate
from src2.util import *
import os
import random
from tqdm import tqdm
import numpy as np
import torch

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def train(args):
    if args.load_trained:
        last_epoch, arch, model, tokenizer, scores = load_checkpoint(
            args.pytorch_dump_path)
    else:
        # May load local file or download from huggingface
        model, tokenizer = load_pretrained_model_tokenizer(args.model_type,
                                                           base_model=args.local_model,
                                                           base_tokenizer=args.local_tokenizer,
                                                           device=args.device)
        last_epoch = 1
    train_dataset = load_data(args.data_path, args.data_name, args.batch_size,
                              tokenizer, split="train", device=args.device, padding=args.padding)
    optimizer = init_optimizer(model, args.learning_rate,
                               args.warmup_proportion, args.num_train_epochs,
                               args.data_size, args.batch_size)

    model.train()
    global_step = 0
    best_score = 0
    for epoch in range(last_epoch, args.num_train_epochs + 1):
        tr_loss = 0
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(tqdm(train_dataset)):
            if batch is None:
                break
            tokens_tensor, segments_tensor, mask_tensor, label_tensor, _, _ = batch
            if args.model_type == "BertForNextSentencePrediction" or args.model_type == "BertForQuestionAnswering":
                loss = model(tokens_tensor, segments_tensor, mask_tensor,
                             label_tensor)
            else:
                loss, logits = model(tokens_tensor, segments_tensor,
                                     mask_tensor, label_tensor)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                best_score = eval_select(model, tokenizer,
                                         args.pytorch_dump_path, best_score,
                                         epoch, args.model_type)

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(model, tokenizer, args.pytorch_dump_path,
                                 best_score, epoch, args.model_type)

    scores = test(args, split="test")
    print_scores(scores)


def eval_select(model, tokenizer, model_path, best_score, epoch, arch):
    scores_dev = test(args, split="dev", model=model, tokenizer=tokenizer,
                      training_or_lm=True)
    print_scores(scores_dev, mode="dev")
    scores_test = test(args, split="test", model=model, tokenizer=tokenizer,
                       training_or_lm=True)
    print_scores(scores_test)

    if scores_dev[1][0] > best_score:
        best_score = scores_dev[1][0]
        model_path = "{}_{}".format(model_path, epoch)
        print("Save PyTorch model to {}".format(model_path))
        save_checkpoint(epoch, arch, model, tokenizer, scores_dev, model_path)

    return best_score


def test(args, split="test", model=None, tokenizer=None, training_or_lm=False):
    if model is None:
        if args.load_trained:
            epoch, arch, model, tokenizer, scores = load_checkpoint(
            args.pytorch_dump_path)
        else:
            # May load local file or download from huggingface
            model, tokenizer = load_pretrained_model_tokenizer(args.model_type,
                                                               base_model=args.local_model,
                                                               base_tokenizer=args.local_tokenizer,
                                                               device=args.device)

    if training_or_lm:
        # Load MB data
        test_dataset = load_data(args.data_path, args.data_name,
                                 args.batch_size, tokenizer, split,
                                 device=args.device, padding=args.padding)
    else:
        # Load Robust04 data
        test_dataset = load_trec_data(args.data_path, args.data_name,
                                      args.batch_size, tokenizer, split,
                                      args.device, padding=args.padding)

    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    f = open(args.output_path, "w")
    predicted = open(args.predict_path, "w")
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
            predicted.write(
                "{} Q0 {} {} {} bert\n".format(qid, docid, lineno, s[1]))
            lineno += 1
        del predictions

    f.close()
    predicted.close()

    map, mrr, p30 = evaluate(args.trec_eval_path,
                             predictions_file=args.predict_path, \
                             qrels_file=os.path.join(args.data_path,
                                                     args.qrels_file))

    torch.cuda.empty_cache()
    model.train()

    return [["map", "mrr", "p30"], [map, mrr, p30]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='[1, 8, 16, 32]')
    parser.add_argument('--data_size', default=41579, type=int,
                        help='[tweet2014: 41579]')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='')
    parser.add_argument('--data_path', default='data', help='')
    parser.add_argument('--data_name', default='robust04',
                        help='[mb, robust04, tfidf_sents]')
    parser.add_argument('--pytorch_dump_path', default='saved.model', help='')
    parser.add_argument('--load_trained', action='store_true', default=False,
                        help='')
    parser.add_argument('--chinese', action='store_true', default=False,
                        help='')
    parser.add_argument('--padding', default=None, help='[None, left, right]')
    parser.add_argument('--trec_eval_path',
                        default='eval/trec_eval.9.0.4/trec_eval', help='')
    parser.add_argument('--local_model', default=None,
                        help='[None, path to local model file]')
    parser.add_argument('--local_tokenizer', default=None,
                        help='[None, path to local vocab file]')
    parser.add_argument('--eval_steps', default=-1, type=int,
                        help='evaluation per [eval_steps] steps, -1 for evaluation per epoch')
    parser.add_argument('--model_type', default='BertForNextSentencePrediction',
                        help='')
    parser.add_argument('--output_path', default='out.tmp', help='')
    parser.add_argument('--predict_path', default='prediction.trec', help='')
    parser.add_argument('--qrels_file', default='qrels.microblog.txt', help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        scores = test(args)
        print_scores(scores)
