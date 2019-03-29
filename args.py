from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='BERT4retrieval')
    parser.add_argument('--target_path', default='../Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar',
                        help='Path to Anserini target jar')
    parser.add_argument('--data_path',
                        default='../Anserini/src/main/resources',
                        help='Path to Anserini resouces')
    parser.add_argument('--index_path', default='/tuna1/indexes/lucene-index.robust04.pos+docvectors+rawdocs',
                        help='Path to Lucene index')
    parser.add_argument('--folds_path',
                        default='../Anserini/src/main/resources/fine_tuning',
                        help='Path to Robust04 folds')
    parser.add_argument('--output_path',
                        default='robust04_bm25_rm3_cv_folder_1.txt',
                        help='Path to write outputs')
    parser.add_argument('--prediction_path',
                        default='predict_robust04_rm3_cv.txt',
                        help='Path to predictions')
    parser.add_argument('--cv_folds', default=2, help='?')

    args = parser.parse_args()
    return args
