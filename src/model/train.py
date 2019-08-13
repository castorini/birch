import numpy as np
import torch

from .test import *
from .utils import init_optimizer, load_checkpoint, print_scores, load_pretrained_model_tokenizer
from .data import *

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def train(args):
    if args.load_trained:
        last_epoch, model, tokenizer, scores = load_checkpoint(args.model_path)
    else:
        # May load local file or download from huggingface
        model, tokenizer = load_pretrained_model_tokenizer(base_model=args.local_model,
                                                           base_tokenizer=args.local_tokenizer,
                                                           device=args.device)
        last_epoch = 1

    train_dataset = DataGenerator(args.data_path, args.collection,
                                  args.batch_size, tokenizer, 'train',
                                  args.device)
    validate_dataset = DataGenerator(args.data_path, args.collection,
                                     args.batch_size, tokenizer, 'dev',
                                     args.device)

    optimizer = init_optimizer(model, args.learning_rate, args.warmup_proportion,
                               args.num_train_epochs, args.batch_size)

    model.train()
    best_score = 0
    step = 0
    for epoch in range(last_epoch, args.num_train_epochs + 1):
        print('Epoch: {}'.format(epoch))
        tr_loss = 0
        while True:
            batch = train_dataset.load_batch()
            if batch is None:
                break
            tokens_tensor, segments_tensor, mask_tensor, label_tensor, _, _ = batch
            loss = model(tokens_tensor, segments_tensor, mask_tensor, label_tensor)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                print('Step: {}'.format(step))
                best_score = eval_select(args, model, tokenizer, validate_dataset,
                                         args.model_path,
                                         best_score, epoch)
            step += 1

        print('[train] loss: {}'.format(tr_loss))
        best_score = eval_select(args, model, tokenizer, validate_dataset,
                                 args.model_path, best_score, epoch)

    scores = test(args, split='test')
    print_scores(scores)
