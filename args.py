from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='BERT4retrieval')
    parser.add_argument('--experiment', default='basic-msmarco',
                        help='Experiment name for logging')
    parser.add_argument('--target_path', default='../Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar',
                        help='Path to Anserini target jar')
    parser.add_argument('--data_path',
                        default='../Anserini/src/main/resources',
                        help='Path to Anserini resouces')
    parser.add_argument('--index_path', default='/tuna1/indexes/lucene-index.robust04.pos+docvectors+rawdocs',
                        help='Path to Lucene index')
    parser.add_argument('--collection', default='robust04', help='[robust04, core17, core18]')
    parser.add_argument('--qrels', default='qrels.robust2004.txt')
    parser.add_argument('--bm25_res', default='robust04_rm3_5cv_sent_fields.txt')
    parser.add_argument('--folds_path',
                        default='../Anserini/src/main/resources/fine_tuning/robust04-paper2-folds.json',
                        help='Path to Robust04 folds')
    parser.add_argument('--output_path',
                        default='robust04_bm25_rm3_cv_folder_1.txt',
                        help='Path to write outputs')
    parser.add_argument('--prediction_path',
                        default='predict.robust04_rm3_cv.txt',
                        help='Path to predictions')
    parser.add_argument('--cv_folds', default=5)

    args, other = parser.parse_known_args()
    return args, other
