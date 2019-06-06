import os

import torch

from inference_utils import evaluate, load_checkpoint, print_scores, load_pretrained_model_tokenizer
from data import load_data


def test(args, predictions_path, model=None, tokenizer=None):
    if model is None:
        if args.load_trained:
            epoch, arch, model, tokenizer, scores = load_checkpoint(
            args.model_path)
        else:
            # May load local file or download from huggingface
            model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                               base_tokenizer=args.local_tokenizer)

    test_dataset = load_data(args.data_path, args.collection,
                                  args.batch_size, tokenizer)

    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    f = open(args.output_path, "w")
    predicted = open(predictions_path, "w")
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
            predicted.write("{} Q0 {} {} {} bert\n".format(qid, docid, lineno, s[1]))
            lineno += 1
            predicted.flush()
        del predictions

    f.close()
    predicted.close()

    eval_path = os.path.join(args.anserini_path, 'eval', 'trec_eval.9.0.4', 'trec_eval')

    map, mrr, p30 = evaluate(eval_path,
                             predictions_file=args.predict_path,
                             qrels_file=os.path.join(args.data_path,
                                                     'topics-and-qrels',
                                                     args.qrels))

    return [["map", "mrr", "p30"], [map, mrr, p30]]
