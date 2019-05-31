import os
import json
from src.utils import parse_doc_from_index, clean_html, tokenizer, MAX_INPUT_LENGTH, chunk_sent
from src.args import get_args

import jnius_config
# TODO: make path dynamic
jnius_config.set_classpath("../Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar")

try:
    from jnius import autoclass
except KeyError:
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-oracle'
    from jnius import autoclass

JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')

def build_searcher(k1=0.9, b=0.4, fb_terms=10, fb_docs=10, original_query_weight=0.5,
                index_path="index/lucene-index.robust04.pos+docvectors+rawdocs", rm3=False):
    searcher = JSearcher(JString(index_path))
    searcher.setBM25Similarity(k1, b)
    if not rm3:
        searcher.setDefaultReranker()
    else:
        searcher.setRM3Reranker(fb_terms, fb_docs, original_query_weight, False)
    return searcher


def anserini_retriever(query, searcher, para_num=20):
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


def search_document(searcher, qid2docid, qid2text, output_fn, collection='core17', K=1000, topics=None):
    qidx, didx = 1, 1
    with open(output_fn, 'w') as out:
        if 'core' in collection:
            # Robust04 provides CV topics
            topics = qid2text
        # print(topics)
        for qid in topics:
            print(qid + ':')
            text = qid2text[qid]
            hits = searcher.search(JString(text), K)
            for i in range(len(hits)):
                sim = hits[i].score
                docno = hits[i].docid
                label = 1 if qid in qid2docid and docno in qid2docid[qid] else 0
                content = hits[i].content
                # print(content)
                if collection == 'core18':
                    content_json = json.loads(content)
                    content = ''
                    for each in content_json['contents']:
                        if each is not None and 'content' in each.keys():
                            content += '{}\n'.format(each['content'])
                if collection == 'robust04':
                    content = parse_doc_from_index(content)
                clean_content = clean_html(content, collection=collection)
                tokenized_content = tokenizer.tokenize(clean_content)
                sentid = 0
                for sent in tokenized_content:
                    # Split sentence if it's longer than BERT's maximum input length
                    if len(sent.strip().split()) > MAX_INPUT_LENGTH:
                        # print('a')
                        seq_list = chunk_sent(sent, MAX_INPUT_LENGTH)
                        for seq in seq_list:
                            # print('s')
                            sentno = docno + '_' + str(sentid)
                            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, text, seq, qid, sentno, qidx, didx))
                            out.flush()
                            sentid += 1
                            didx += 1
                    else:
                        # print('b')
                        sentno = docno + '_' + str(sentid)
                        out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, text, sent, qid, sentno, qidx, didx))
                        out.flush()
                        sentid += 1
                        didx += 1
                    # print('{} {}'.format(didx, qidx))
            qidx += 1


if __name__ == '__main__':
    args, _ = get_args()

    searcher = build_searcher()
    query = "Radio Waves and Brain Cancer"
    ret = anserini_retriever(query, searcher)
    print(query)
    print(ret)

