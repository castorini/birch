import argparse

def get_args():
    parser = ArgumentParser(description='BERT4retrieval')
    parser.add_argument('--data_path',
                        default='../Anserini/src/main/resources/topics-and-qrels',
                        help='Path to Anserini resources that contains Robust04 data')

    args = parser.parse_args()
    return args