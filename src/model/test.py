import torch

from .eval import evaluate
from .utils import print_scores, load_checkpoint, save_checkpoint, load_pretrained_model_tokenizer
from .data import *


def eval_select(args, model, tokenizer, validate_dataset, model_path, best_score, epoch):
    scores_dev = test(args, split='dev', model=model, test_dataset=validate_dataset)
    print_scores(scores_dev, mode='dev')

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
            epoch, model, tokenizer, scores = load_checkpoint(args.model_path, device=args.device)
        else:
            model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                               base_tokenizer=args.local_tokenizer,
                                                               device=args.device)

        assert test_dataset is None
        print('Loading {} set...'.format(split))
        test_dataset = DataGenerator(args.data_path, '{}_sents'.format(args.collection), args.batch_size, tokenizer, split, args.device)

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
        if args.collection == 'mb':
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
        else:  # robust04 or core17/18
            qids = qid_tensor.cpu().detach().numpy()
            docids = docid_tensor.cpu().detach().numpy()
            assert len(qids) == len(predicted_index)
            for p, l, s, qid, docid in zip(predicted_index, label_batch, scores, qids, docids):
                predict_file.write('{} Q0 {} {} {} bert\n'.format(qid, docid, line_no, s[1]))
                line_no += 1
        predict_file.flush()

        label_new = label_new if len(label_new) > 0 else label_batch
        predicted_index_new = predicted_index_new if len(predicted_index_new) > 0 else predicted_index
        labels.extend(label_new)
        prediction_index_list += predicted_index_new
        del predictions

    output_file.close()
    predict_file.close()
    torch.cuda.empty_cache()
    model.train()

    map, p20, ndcg20 = evaluate(args.trec_eval_path, predictions_file=args.predict_path, \
                            qrels_file=os.path.join(args.data_path, 'qrels', 'qrels.{}.txt'.format(args.collection)))
    return [['map', 'p20', 'ndcg20'], [map, p20, ndcg20]]
