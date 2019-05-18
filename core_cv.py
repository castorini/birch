from args import get_args
from utils import *

from searcher import *

if __name__ == '__main__':
    args = get_args()
    collection = args.collection
    data_path = args.data_path
    index_path = args.index_path  # '/tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs'
    output_fn = args.output_path  # 'core17.csv'

    fqrel = os.path.join(data_path, 'topics-and-qrels', 'qrels.' + collection + '.txt')
    ftopic = os.path.join(data_path, 'topics-and-qrels', 'topics.' + collection + '.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic, collection=collection)

    searcher = build_searcher(k1=0.9, b=0.4, index_path=index_path, rm3=True)
    search_core(searcher, qid2docid, qid2text, output_fn, collection, K=1000)