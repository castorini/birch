from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='birch')
    parser.add_argument('--mode', default='retrieval', help='[training, inference, retrieval]')
    parser.add_argument('--output_path', default='out.tmp', help='Name of log file')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--collection', default='robust04', help='[mb, robust04, core17, core18]')

    # Interactive
    parser.add_argument('--interactive', action='store_true', default=False, help='Batch evaluation if not set')
    parser.add_argument('--query', default='hubble space telescope', help='Query string')
    parser.add_argument('--interactive_path', default='data/datasets/query_sents.csv', help='Path to output sentence results from query')

    # Retrieval
    parser.add_argument('--experiment', default='base_mb_robust04', help='Experiment name for logging')
    parser.add_argument('--anserini_path', default='../Anserini', help='Path to Anserini root')
    parser.add_argument('--index_path', default='lucene-index.robust04.pos+docvectors+rawdocs', help='Path to Lucene index')
    parser.add_argument('--cv_fold', default=5)

    # Training
    parser.add_argument('--device', default='cpu', help='[cuda, cpu]')
    parser.add_argument('--trec_eval_path', default='eval/trec_eval.9.0.4/trec_eval')
    parser.add_argument('--model_path', default='models/saved.tmp', help='Path to pretrained model')
    parser.add_argument('--predict_path', default='predict.tmp')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int, help='Number of evaluation steps, -1 for evaluation per epoch')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of training to perform linear learning rate warmup. E.g., 0.1 = 10%% of training.')
    parser.add_argument('--local_model', default=None, help='[None, path to local model file]')
    parser.add_argument('--local_tokenizer', default=None, help='[None, path to local vocab file]')
    parser.add_argument('--load_trained', action='store_true', default=False, help='Load pretrained model if True')

    args, other = parser.parse_known_args()
    return args, other
