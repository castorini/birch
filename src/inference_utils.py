import torch
from pytorch_pretrained_bert import BertTokenizer, \
    BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM

def load_pretrained_model_tokenizer(base_model=None, base_tokenizer=None):
    # Load pre-trained model (weights)
    if base_model is None:
        # Download from huggingface
        base_model = "bert-base-uncased"
    model = BertForNextSentencePrediction.from_pretrained(base_model)

    if base_tokenizer is None:
        # Download from huggingface
        tokenizer = BertTokenizer.from_pretrained(base_model)
    else:
        # Load local vocab file
        tokenizer = BertTokenizer.from_pretrained(base_tokenizer)
    return model, tokenizer


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
    state = torch.load(filename, map_location='cpu')
    return state['epoch'], state['arch'], state['model'], state['tokenizer'], \
           state['scores']


def print_scores(scores, mode="test"):
    print("")
    print("[{}] ".format(mode), end="")
    for sn, score in zip(scores[0], scores[1]):
        print("{}: {}".format(sn, score), end=" ")
    print("")
