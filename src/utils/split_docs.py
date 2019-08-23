import os
import sys

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)

from utils.doc_utils import *
from utils.searcher import *
from args import get_args
from shutil import copyfileobj

if __name__ == '__main__':
    args, _ = get_args()
    collection = args.collection
    anserini_path = args.anserini_path
    data_path = args.data_path
    index_path = args.index_path
    output_fn = os.path.join(args.data_path, 'datasets', collection + '_sents.csv')

    fqrel = os.path.join(data_path, 'qrels', 'qrels.' + collection + '.txt')
    ftopic = os.path.join(data_path, 'topics', 'topics.' + collection + '.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic, collection=collection)

    docsearch = Searcher(anserini_path)

    if collection == 'robust04':
        with open(os.path.join(data_path, 'folds', collection + '-folds.json')) as f:
            folds = json.load(f)
        params = ["0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 26 8 0.30"]
        folder_idx = 1
        for topics, param in zip(folds, params):
            # Extract each parameter
            k1, b, fb_terms, fb_docs, original_query_weight = map(float, param.strip().split())
            searcher = docsearch.build_searcher(k1=k1, b=b, fb_terms=fb_terms,
                                                fb_docs=fb_docs, original_query_weight=original_query_weight,
                                                index_path=index_path, rm3=True)
            docsearch.search_document(searcher, qid2docid, qid2text,
                                      output_fn + str(folder_idx),
                                      collection, 1000, topics)

            folder_idx += 1

        with open(output_fn, 'w') as outfile:
            for infile in [output_fn + str(n) for n in range(1, folder_idx)]:
                copyfileobj(open(infile), outfile)

        for i in range(5):
            os.remove(output_fn + str(i + 1))

    else:
        searcher = docsearch.build_searcher(k1=0.9, b=0.4, index_path=index_path, rm3=True)
        docsearch.search_document(searcher, qid2docid, qid2text, output_fn, collection, K=1000)
