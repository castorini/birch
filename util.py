import torch
from pytorch_pretrained_bert import BertTokenizer, \
    BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam


def load_pretrained_model_tokenizer(model_type="BertForSequenceClassification",
                                    base_model=None, base_tokenizer=None,
                                    device="cuda", chinese=False):
    # Load pre-trained model (weights)
    if base_model is None:
        # Download from huggingface
        if chinese:
            base_model = "bert-base-chinese"
        else:
            base_model = "bert-base-uncased"
    if model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(base_model)
        # Load pre-trained model tokenizer (vocabulary)
    elif model_type == "BertForNextSentencePrediction":
        model = BertForNextSentencePrediction.from_pretrained(base_model)
    elif model_type == "BertForMaskedLM":
        model = BertForMaskedLM.from_pretrained(base_model)
    else:
        print("[Error]: unsupported model type")
        return None, None

    if base_tokenizer is None:
        # Download from huggingface
        tokenizer = BertTokenizer.from_pretrained(base_model)
    else:
        # Load local vocab file
        tokenizer = BertTokenizer.from_pretrained(base_tokenizer)
    model.to(device)
    return model, tokenizer


def init_optimizer(model, learning_rate, warmup_proportion, num_train_epochs,
                   data_size, batch_size):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = data_size / batch_size * num_train_epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay],
         'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)

    return optimizer


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
    return state['epoch'], state['arch'], state['model'], state['tokenizer'], \
           state['scores']


def print_scores(scores, mode="test"):
    print("")
    print("[{}] ".format(mode), end="")
    for sn, score in zip(scores[0], scores[1]):
        print("{}: {}".format(sn, score), end=" ")
    print("")
