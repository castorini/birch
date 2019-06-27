from utils import *
from searcher import *
from shutil import copyfileobj
from args import get_args

if __name__ == '__main__':
    args, _ = get_args()
    anserini_path = args.anserini_path
    index_path = args.index_path
    cv_fold = args.cv_fold
    output_fn = os.path.join(args.data_path, 'datasets', "robust04_" + str(cv_fold) + "cv")

    # TODO: dynamic params
    if cv_fold == '5':
        folds_file = "robust04-paper2-folds.json"
        params = ["0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 47 9 0.30",
                  "0.9 0.5 26 8 0.30"]
    else:
        folds_file = "robust04-paper1-folds.json"
        params = ["0.9 0.5 50 17 0.20",
                  "0.9 0.5 26 8 0.30"]

    folds_path = os.path.join(anserini_path, 'src', 'main', 'resources', 'fine_tuning', folds_file)

    fqrel = os.path.join(anserini_path, 'src', 'main', 'resources', 'topics-and-qrels', 'qrels.robust2004.txt')
    ftopic = os.path.join(anserini_path, 'src', 'main', 'resources', 'topics-and-qrels', 'topics.robust04.301-450.601-700.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic, collection='robust04')

    with open(os.path.join(folds_path)) as f:
        folds = json.load(f)

    folder_idx = 1
    docsearch = Searcher(anserini_path)
    for topics, param in zip(folds, params):
        print(folder_idx)
        # Extract each parameter
        k1, b, fb_terms, fb_docs, original_query_weight = map(float, param.strip().split())
        searcher = docsearch.build_searcher(k1=k1, b=b, fb_terms=fb_terms, fb_docs=fb_docs,
          original_query_weight=original_query_weight,index_path=index_path, rm3=True)
        docsearch.search_document(searcher, qid2docid, qid2text, output_fn + str(folder_idx),
          'robust04', 1000, topics)

        folder_idx += 1

    with open(output_fn + ".csv", 'w') as outfile:
        for infile in [output_fn + str(n) for n in range(1, folder_idx)]:
            copyfileobj(open(infile), outfile)
