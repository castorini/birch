from tqdm import tqdm
import random 
import os 
import numpy as np
import argparse

import torch

from util import *

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def train(args):
    model, tokenizer = load_pretrained_model_tokenizer(args.model_type, device=args.device)
    train_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "train", args.device)
    validate_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "validate", args.device)
    test_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, "test", args.device)
    optimizer = init_optimizer(model, args.learning_rate, args.warmup_proportion, args.num_train_epochs, len(train_dataset))
    model_path = os.path.join(args.pytorch_dump_path, "{}_finetuned.pt".format(args.data_name))
    
    model.train()
    global_step = 0
    best_score = 0
    for epoch in range(1, args.num_train_epochs+1):
        tr_loss = 0
        random.shuffle(train_dataset)
        for step, batch in enumerate(tqdm(train_dataset)):
            tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch
            if args.model_type == "BertForNextSentencePrediction" or args.model_type == "BertForQuestionAnswering":
                loss = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            else:
                loss, logits = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            
            if args.eval_steps > 0 and step % args.eval_steps == 0:
                best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score)

        print("[train] loss: {}".format(tr_loss))
        best_score = eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score)

    acc_test, p1_test = test(args, split="test")
    print("[test]: acc: {}, p@1: {}".format(acc_test, p1_test))

def eval_select(model, tokenizer, validate_dataset, test_dataset, model_path, best_score):
    print("")
    acc_dev, p1_dev, pre_dev, rec_dev, f1_dev = test(args, split="validate", model=model, tokenizer=tokenizer, test_dataset=validate_dataset)
    print("[dev] acc: {}, p@1: {}, precision: {}, recall: {}, f1: {}".format(acc_dev, p1_dev, pre_dev, rec_dev, f1_dev))
    acc_test, p1_test, pre_test, rec_test, f1_test = test(args, split="test", model=model, tokenizer=tokenizer, test_dataset=test_dataset)
    print("[test] acc: {}, p@1: {}, precision: {}, recall: {}, f1: {}".format(acc_test, p1_test, pre_test, rec_test, f1_test))
    
    if acc_dev > best_score:
        best_score = acc_dev
        # Save pytorch-model
        print("Save PyTorch model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)

    return best_score

def test(args, split="test", model=None, tokenizer=None, test_dataset=None):
    if model is None:
        model_path = os.path.join(args.pytorch_dump_path, "{}_finetuned.pt".format(args.data_name))
        print("Load PyTorch model from {}".format(model_path))
        model, tokenizer = load_pretrained_model_tokenizer(args.model_type, device=args.device)
        model.load_state_dict(torch.load(model_path))
    if test_dataset is None: 
        print("Load test set")
        test_dataset = load_data(args.data_path, args.data_name, args.batch_size, tokenizer, split, args.device)
    
    model.eval()
    prediction_score_list, prediction_index_list, labels = [], [], []
    for tokens_tensor, segments_tensor, mask_tensor, label_tensor in test_dataset:
        predictions = model(tokens_tensor, segments_tensor, mask_tensor)
        predicted_index = list(torch.argmax(predictions, dim=1).cpu().numpy())
        prediction_index_list += predicted_index
        predicted_score = list(predictions[:, 1].cpu().detach().numpy())
        prediction_score_list.extend(predicted_score)
        labels.extend(list(label_tensor.cpu().detach().numpy()))
        
        del predictions

    acc = get_acc(prediction_index_list, labels)
    p1 = get_p1(prediction_score_list, labels, args.data_path, args.data_name, split)
    pre, rec, f1 = get_pre_rec_f1(prediction_index_list, labels)

    torch.cuda.empty_cache()
    model.train()
    
    return acc, p1, pre, rec, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='[train, test]')
    parser.add_argument('--device', default='cuda', help='[cuda, cpu]')
    parser.add_argument('--batch_size', default=16, type=int, help='[1, 8, 16, 32]')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='')
    parser.add_argument('--data_path', default='/data/wyang/ShortTextSemanticSimilarity/data/corpora/', help='')
    parser.add_argument('--data_name', default='annotation', help='annotation or youzan_new')
    parser.add_argument('--pytorch_dump_path', default='model/', help='')
    parser.add_argument('--eval_steps', default=-1, type=int, help='evaluation per [eval_steps] steps, -1 for evaluation per epoch')
    parser.add_argument('--model_type', default='BertForNextSentencePrediction', help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        acc_test, p1_test = test(args)
        print("[test]: acc: {}, p@1: {}".format(acc_test, p1_test))
