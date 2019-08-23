import os
import sys

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)

from utils.doc_utils import *
from utils.searcher import *
from args import get_args

if __name__ == '__main__':
    args, _ = get_args()
    collection = args.collection
    anserini_path = args.anserini_path
    data_path = args.data_path
    index_path = args.index_path
    output_fn = os.path.join(args.data_path, 'datasets', collection + '_5cv.csv')

    fqrel = os.path.join(data_path, 'qrels', 'qrels.' + collection + '.txt')
    ftopic = os.path.join(data_path, 'topics', 'topics.' + collection + '.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic, collection=collection)

    docsearch = Searcher(anserini_path)
    searcher = docsearch.build_searcher(k1=0.9, b=0.4, index_path=index_path, rm3=True)
    docsearch.search_document(searcher, qid2docid, qid2text, output_fn, collection, K=1000)
