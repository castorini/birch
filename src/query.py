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
    return docsearch.search_query(searcher, query, output_fn, collection, K=1000)


def visualize_scores(collection_path, bert_scores):
    top_rank_docs = []
    with open(collection_path, 'r', encoding="utf-8") as collection:
        for line in collection:
            label, doc_score, query, sent, qid, did, qno, dno = line.strip().split('\t')
            doc_id, sent_id = did.split('_')[0], int(did.split('_')[1])
            bert_score = bert_scores['0'][doc_id][sent_id]
            top_rank_docs.append((did, float(doc_score), bert_score, float(doc_score) - bert_score))

    top_rank_docs.sort(key=lambda x: x[3], reverse=True)
    return top_rank_docs


if __name__ == '__main__':
    args, _ = get_args()
    docid2text = query_sents(args)
