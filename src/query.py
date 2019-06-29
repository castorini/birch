from utils import *
from searcher import *
from args import get_args


def query_sents(args):
    collection = args.collection
    anserini_path = args.anserini_path
    index_path = args.index_path
    query = args.query
    output_fn = args.interactive_path

    docsearch = Searcher(anserini_path)
    searcher = docsearch.build_searcher(k1=0.9, b=0.4, index_path=index_path,
                                        rm3=True)
    docsearch.search_query(searcher, query, output_fn, collection, K=1000)


if __name__ == '__main__':
    args, _ = get_args()
    query_sents(args)
