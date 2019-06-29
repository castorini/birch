import os

import torch

from .eval import evaluate
from .utils import print_scores, load_checkpoint, save_checkpoint, load_pretrained_model_tokenizer
from .data import load_data, load_trec_data


def eval_select(args, model, tokenizer, best_score, epoch):
    scores_dev = test(args, split='dev', model=model, tokenizer=tokenizer, training_or_lm=True)
    print_scores(scores_dev, mode='dev')
    scores_test = test(args, split='test', model=model, tokenizer=tokenizer, training_or_lm=True)
    print_scores(scores_test,  mode='test')

    if scores_dev[1][0] > best_score:
        best_score = scores_dev[1][0]
        model_path = '{}_{}'.format(args.model_path, epoch)
        save_checkpoint(epoch, model, tokenizer, scores_dev, model_path)

    return best_score


def test(args, split='test', model=None, tokenizer=None, training_or_lm=False):
    if model is None:
        if args.load_trained:
            epoch, model, tokenizer, scores = load_checkpoint(args.model_path, args.device)
        else:
            # May load local file or download from huggingface
            model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                               base_tokenizer=args.local_tokenizer,
                                                               device=args.device)

    if training_or_lm:
        # Load MB data
        test_dataset = load_data(args.data_path, args.collection,
                                 args.batch_size, tokenizer, split,
                                 device=args.device)
    else:
        # Load Robust04 data
        collection = args.collection if not args.interactive else 'query_sents'
        test_dataset = load_trec_data(args.data_path, collection,
                                      args.batch_size, tokenizer, split,
                                      args.device)

    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    output_file = open(os.path.join('log', args.output_path), 'w')
    predictions_path = os.path.join(args.data_path, 'predictions', args.predict_path)
    predict_file = open(predictions_path, 'w')
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
            output_file.write('{}\t{}\n'.format(lineno, p))
            predict_file.write('{} Q0 {} {} {} bert\n'.format(qid, docid, lineno, s[1]))
            lineno += 1
            predict_file.flush()
        del predictions

    output_file.close()
    predict_file.close()

    if args.interactive:
        return None

    else:
        map, mrr, p30 = evaluate(args.trec_eval_path,
                                 predictions_file=predictions_path,
                                 qrels_file=os.path.join(args.data_path, args.qrels_file))

        return [['map', 'mrr', 'p30'], [map, mrr, p30]]
