import tqdm import tqdm
import random 
import os 
import numpy as np

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

from util import *


def train(args):
    model, tokenizer = load_pretrained_model_tokenizer(device=args.device)
    train_dataset = load_data(args.data_path, args.data_name, args.batch_size, args.split, args.device)
    optimizer = init_optimizer(model, args.learning_rate, args.warmup_proportion, args.num_train_epochs)
    
    model.train()
    global_step = 0
    best_score = 0
    for epoch in range(1, num_train_epochs+1):
	tr_loss = 0
	random.shuffle(train_data_set)
	for step, batch in enumerate(tqdm(train_data_set)):
	    tokens_tensor, segments_tensor, mask_tensor, label_tensor = batch
	    # Predict all tokens
	    loss, logits = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
	    loss.backward()
	    tr_loss += loss.item()
	    optimizer.step()
	    model.zero_grad()
	    global_step += 1
        
        acc_dev, p1_dev = test(args, split="dev", model=model)
        print("[dev]: acc: {}, p@1: {}".format(acc_dev, p1_dev))
        acc_test, p1_test = test(args, split="test", model=model)
        print("[test]: acc: {}, p@1: {}".format(acc_test, p1_test))
        
        if p1_dev > best_score:
            best_score = p1_dev
            # Save pytorch-model
            print("Save PyTorch model to {}".format(pytorch_dump_path))
            torch.save(model.state_dict(), os.join(args.pytorch_dump_path, "{}_finetuned.pt"))

    acc_test, p1_test = test(args, split="test", model=model)
    print("[test]: acc: {}, p@1: {}".format(acc_test, p1_test))

def test(args, split="test", model=None):
    if model is None:
        pass
    
    model.eval()
    test_dataset = load_data(args.data_path, args.data_name, args.batch_size, split, args.device)
    prediction_score_list, prediction_index_list, labels = [], [], []
    
    for tokens_tensor, segments_tensor, mask_tensor, label_tensor in test_dataset:
        predictions = model(tokens_tensor, segments_tensor, mask_tensor)
        predictions = model(tokens_tensor, segments_tensor)
        predicted_index = list(torch.argmax(predictions, dim=1).cpu().numpy())
        prediction_index_list += predicted_index
        predicted_score = list(predictions[:, 1].cpu().detach().numpy())
        prediction_score_list.extend(predicted_score)
        labels.extend(list(label_tensor.cpu().detach().numpy()))
    
    acc = get_acc(prediction_index_list, labels)
    p1 = get_p1(prediction_score_list, labels, args.data_path, args.data_name, args.split)
    return acc, p1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', default=16, type=int, help='suggested value: 1 or 16 or 32')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='')
    parser.add_argument('--data_path', default='/data/wyang/ShortTextSemanticSimilarity/data/corpora/', help='')
    parser.add_argument('--data_name', default='annotation', help='annotation or youzan_new')
    parser.add_argument('--pytorch_dump_path', default='model/', help='')

    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    args = parser.parse_args()
    
    train(args)
