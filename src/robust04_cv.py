import sys
reload(sys)

from utils import *
from searcher import *
from args import get_args

if __name__ == '__main__':
    args, _ = get_args()
    collection = args.collection
    anserini_path = args.anserini_path
    index_path = args.index_path
    output_fn = args.output_path
    folds_path = args.folds_path
    cv_fold = args.cv_fold

    fqrel = os.path.join(anserini_path, 'src', 'main', 'resources', 'topics-and-qrels', 'qrels.' + collection + '.txt')
    ftopic = os.path.join(anserini_path, 'src', 'main', 'resources', 'topics-and-qrels', 'topics.' + collection + '.301-450.601-700.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic, collection='robust04')

    # TODO: dynamic params
    if cv_fold == 5:
        with open(os.path.join(folds_path, 'robust04-paper2-folds.json')) as f:
            folds = json.load(f)
        params = ["0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 26 8 0.30"]
    else:
        with open(os.path.join(folds_path, 'robust04-paper1-folds.json')) as f:
            folds = json.load(f)
        params = ["0.9 0.5 50 17 0.20",
                  "0.9 0.5 26 8 0.30"]

    folder_idx = 1
    for topics, param in zip(folds, params):
        print(folder_idx)
        # Extract each parameter
        k1, b, fb_terms, fb_docs, original_query_weight = map(float, param.strip().split())
        searcher = build_searcher(k1=k1, b=b, fb_terms=fb_terms, fb_docs=fb_docs,
                                  original_query_weight=original_query_weight,
                                  index_path=index_path, rm3=True)
        search_document(searcher, qid2docid, qid2text,
                                            output_fn + str(folder_idx),
                                             'robust04', 1000, topics)

        folder_idx += 1
