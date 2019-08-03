from utils import *
from searcher import *
from args import get_args

if __name__ == '__main__':
    args, _ = get_args()
    collection = args.collection
    anserini_path = args.anserini_path
    index_path = args.index_path
    output_fn = args.output_path

    qid2text = {}
    with open('msmarco/msmarco-test2019-queries.tsv', 'r') as query_file:
        for line in query_file:
            qid, query = line.strip().split('\t')
            qid2text[qid] = query

    docsearch = Searcher(anserini_path)
    searcher = docsearch.build_searcher(k1=3.44, b=0.87, index_path=index_path, rm3=True)  # TODO: what to do about the parameters?
    docsearch.search_document(searcher, None, qid2text, output_fn, collection, K=1000)
