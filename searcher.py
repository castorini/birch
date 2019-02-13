import os
import numpy as np

import jnius_config
jnius_config.set_classpath("../target/anserini-0.3.1-SNAPSHOT-fatjar.jar")

from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')

def build_searcher(k1=0.9, b=0.4, index_path="index/lucene-index.robust04.pos+docvectors+rawdocs", rm3=False, chinese=False):
    searcher = JSearcher(JString(index_path))
    searcher.setBM25Similarity(k1, b)
    if not rm3:
        searcher.setDefaultReranker()
    else:
        searcher.setRM3Reranker()
    return searcher

def answerini_retriever(query, searcher, para_num=20):
    try:
        hits = searcher.search(JString(query), para_num)
    except ValueError as e:
        print("Search failure: ", query.encode("utf-8"), e)
        return []

    documents = []

    for document in hits:
        # TODO: filter out impossible sentences
        document_dict = {'text': document.content,
                          'document_score': document.score,
                          'docid': document.docid}
        documents.append(document_dict)

    return documents

if __name__ == '__main__':
    searcher = build_searcher()
    query = "Radio Waves and Brain Cancer"
    ret = answerini_retriever(query, searcher)
    print(query)
    print(ret)
