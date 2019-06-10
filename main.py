# coding=utf-8

import random
import argparse

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
        last_epoch, arch, model, tokenizer, scores, _, step = load_checkpoint(args.pytorch_dump_path)
    else:
        model, tokenizer = load_pretrained_model_tokenizer(args.model_type, base_model=args.base_model,
                                                           base_tokenizer=args.base_tokenizer,
                                                           device=args.device, chinese=args.chinese,
                                                           num_labels=args.num_labels)
        step = 0
        last_epoch = 1

    train_dataset = DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, "train", args.device,

    model.train()
    print('training')
    for epoch in range(last_epoch, args.num_train_epochs + 1):
        print("epoch {} ............".format(epoch))
        tr_loss = 0
        # random.shuffle(train_dataset)
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
            global_step += 1

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                print("step: {}".format(step))
                best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path,
                                         best_score, epoch, args.model_type)

            step += 1

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path, best_score,
                                 epoch, args.model_type)
                                         best_score, epoch, args.model_type, step)

            step += 1
            # del tokens_tensor
            # del segments_tensor
            # del mask_tensor
            # del label_tensor

        step = 0

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, args.pytorch_dump_path, best_score,
                                 epoch, args.model_type, step)

    scores = test(args, split="test")
    print_scores(scores)


def eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score, epoch, arch, step):
    scores_dev = test(args, split="dev", model=model, tokenizer=tokenizer, test_dataset=validate_dataset)
    print_scores(scores_dev, mode="dev")
    scores_test = test(args, split="test", model=model, tokenizer=tokenizer, test_dataset=test_dataset)
    print_scores(scores_test)

    if scores_dev[1][0] > best_score:
        best_score = scores_dev[1][0]
        # Save pytorch-model
        model_path = "{}_{}".format(model_path, epoch)
        print("Save PyTorch model to {}".format(model_path))
        save_checkpoint(epoch, arch, model, tokenizer, scores_dev, model_path, test_dataset.label_map, step)

    return best_score

def test(args, split="test", model=None, tokenizer=None, test_dataset=None):
    if model is None:
        epoch, arch, model, tokenizer, scores, label_map = load_checkpoint(args.pytorch_dump_path)
        assert test_dataset is None
        print("Load {} set".format(split))
        test_dataset = DataGenerator(args.data_path, args.data_name, args.batch_size, tokenizer, split, args.device,
                                     args.data_format, label_map=label_map)

    with torch.no_grad():
        model.eval()
        prediction_score_list, prediction_index_list, labels = [], [], []
        f = open(args.output_path, "w")
        pf = open(args.predict_path, "w")
        # qrelf = open(split + '.' + args.qrels_path, "w")

        lineno = 1
        label_map_reverse = {}
        for k in test_dataset.label_map:
            label_map_reverse[test_dataset.label_map[k]] = k
        qid_tensor, docid_tensor = None, None
        while True:
            batch = test_dataset.load_batch()
            if batch is None:
                break
            if len(batch) == 6:
                tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor = batch
            elif len(batch) == 5:
                tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor = batch
            else:
                tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch
            # del batch
            predictions = model(tokens_tensor, segments_tensor, mask_tensor)
            scores = predictions.cpu().detach().numpy()
            predicted_index = list(torch.argmax(predictions, dim=-1).cpu().numpy())
            if args.data_format == "glue" or args.data_format == "regression":
                predicted_score = list(predictions[:, 0].cpu().detach().numpy())
            else:
                predicted_score = list(predictions[:, 1].cpu().detach().numpy())
            prediction_score_list.extend(predicted_score)
            label_batch = list(label_tensor.cpu().detach().numpy())
            label_new = []
            predicted_index_new = []
            if args.data_format == "trec":
                qids = qid_tensor.cpu().detach().numpy()
                if docid_tensor is not None:
                    docids = docid_tensor.cpu().detach().numpy()
                else:
                    docids = list(range(lineno, lineno + len(label_batch)))
                for p, qid, docid, s, label in zip(predicted_index, qids, docids, \
                                                   scores, label_batch):
                    f.write("{}\t{}\n".format(lineno, p))
                    pf.write("{} Q0 {} {} {} bert\n".format(qid, docid, lineno, s[1]))
                    # qrelf.write("{} Q0 {} {}\n".format(qid, docid, label))
                    lineno += 1
            elif args.data_format == "ontonote":
                tokens = tokens_tensor.cpu().detach().numpy()
                for token, p, label in zip(tokens, predicted_index, label_batch):
                    assert len(token) == len(p)
                    assert len(token) == len(label)
                    predicted_index_tmp = []
                    label_tmp = []
                    for a, b, c in zip(token, p, label):
                        a = tokenizer.convert_ids_to_tokens([a])[0]
                        if a == "[SEP]":
                            f.write("\n")
                            break
                        predicted_index_tmp.append(b)
                        label_tmp.append(c)
                        b = label_map_reverse[b]
                        c = label_map_reverse[c]
                        f.write("{} {} {}\n".format(a, b, c))
                    predicted_index_new.append(predicted_index_tmp)
                    label_new.append(label_tmp)
            elif args.data_format == "robust04":
                qids = qid_tensor.cpu().detach().numpy()
                docids = docid_tensor.cpu().detach().numpy()
                assert len(qids) == len(predicted_index)
                for p, l, s, qid, docid in zip(predicted_index, label_batch, scores, qids, docids):
                    f.write("{} Q0 {} {} {} bert {}\n".format(qid, docid, lineno, s[1], l))
                    lineno += 1
            else:
                if qid_tensor is None:
                    qids = list(range(lineno, lineno + len(label_batch)))
                else:
                    qids = qid_tensor.cpu().detach().numpy()
                assert len(qids) == len(predicted_index)
                for qid, p, l in zip(qids, predicted_index, label_batch):
                    f.write("{},{},{}\n".format(qid, p, l))

            label_new = label_new if len(label_new) > 0 else label_batch
            predicted_index_new = predicted_index_new if len(predicted_index_new) > 0 else predicted_index
            labels.extend(label_new)
            prediction_index_list += predicted_index_new
            # del predictions

    f.close()
    pf.close()

    torch.cuda.empty_cache()

    model.train()

    if args.data_format == "trec":
        map, mrr, p30 = evaluate_trec(predictions_file=args.output_path2, \
                                      qrels_file=split + '.' + args.qrels_path)
        return [["map", "mrr", "p30"], [map, mrr, p30]]
    elif args.data_format == "glue" or args.data_format == "regression":
        pearson_r, spearman_r = evaluate_glue(prediction_score_list, labels)
        return [["pearson_r", "spearman_r"], [pearson_r, spearman_r]]
    elif args.data_format == "ontonote":
        acc, pre, rec, f1 = evaluate_ner(prediction_index_list, labels, test_dataset.label_map)
    else:
        acc, pre, rec, f1 = evaluate_classification(prediction_index_list, labels)
    return [["f1", "acc", "precision", "recall"], [f1, acc, pre, rec]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--padding', default=None, help='[None, left, right]')
    parser.add_argument('--batch_size', default=8, type=int, help='[1, 8, 16, 32]')
    parser.add_argument('--data_size', default=41579, type=int, help='[tweet2014: 41579]')
    parser.add_argument('--learning_rate', default=3e-6, type=float, help='')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='')
    parser.add_argument('--data_path', default='data', help='')
    parser.add_argument('--data_name', default='mb', help='mb, robust04')
    parser.add_argument('--pytorch_dump_path', default='saved.model', help='')
    parser.add_argument('--base_model', default=None, help='[None, path to local file]')
    parser.add_argument('--base_tokenizer', default=None,
                        help='[None, path to local file]')
    parser.add_argument('--load_trained', action='store_true', default=False, help='')
    parser.add_argument('--chinese', action='store_true', default=False, help='')
    parser.add_argument('--eval_steps', default=-1, type=int,
                        help='evaluation per [eval_steps] steps, -1 for evaluation per epoch')
    parser.add_argument('--model_type', default='BertForNextSentencePrediction', help='')
    parser.add_argument('--output_path', default='predict.tmp', help='')
    parser.add_argument('--predict_path', default='predict.trec', help='')
    parser.add_argument('--qrels_path', default='qrels.trec', help='')
    parser.add_argument('--num_labels', default=2, type=int, help='')
    parser.add_argument('--data_format', default='trec', help='[classification, trec, robust04]')
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        scores = test(args)
        print_scores(scores)
