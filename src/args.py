from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='birch')
    parser.add_argument('--experiment', default='base_mb_robust04',
                        help='Experiment name for logging')
    parser.add_argument('--anserini_path',
                        default='../Anserini',
                        help='Path to Anserini root')
    parser.add_argument('--data_path', default='data', help='')
    parser.add_argument('--index_path', default='/tuna1/indexes/lucene-index.robust04.pos+docvectors+rawdocs',
                        help='Path to Lucene index')
    parser.add_argument('--collection', default='robust04', help='[robust04, core17, core18]')
    parser.add_argument('--qrels_file', default='qrels.robust2004.txt')
    parser.add_argument('--bm25_file', default='robust04_rm3_5cv_sent_fields.txt')
    parser.add_argument('--folds_file',
                        default='robust04-paper2-folds.json',
                        help='Path to Robust04 folds')
    parser.add_argument('--output_path',
                        default='robust04_bm25_rm3_cv_folder_1.txt',
                        help='Path to write outputs')
    parser.add_argument('--cv_fold', default=5)

    parser.add_argument('--run_inference', action='store_true', default=False,
                        help='Evaluate model if True, use prediction files otherwise')
    parser.add_argument('--model_path', default='models/saved.tmp', help='Path to pretrained model')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='[1, 8, 16, 32]')
    parser.add_argument('--local_model', default=None,
                        help='[None, path to local model file]')
    parser.add_argument('--local_tokenizer', default=None,
                        help='[None, path to local vocab file]')
    parser.add_argument('--load_trained', action='store_true', default=False,
                        help='Load pretrained BERT if True')

    args, other = parser.parse_known_args()
    return args, other
