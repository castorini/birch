from utils import *
from searcher import *
from args import get_args
from utils import parse_doc_from_index, clean_html, tokenizer, MAX_INPUT_LENGTH, chunk_sent

import os
import gzip
import csv


# Taken from the TREC DL MS MARCO document ranking script
def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        "Looking for {docid}, found {line}"
    return line.rstrip()


if __name__ == '__main__':
    args, _ = get_args()
    collection = args.collection
    data_path = os.path.join(args.data_path, 'trec_dl')
    anserini_path = args.anserini_path
    index_path = args.index_path
    output_fn = args.output_path

    retrieved = True

    qid2text = {}
    with open(os.path.join(data_path, 'msmarco-test2019-queries.tsv'), 'r') as query_file:
        for line in query_file:
            qid, query = line.strip().split('\t')
            qid2text[qid] = query

    if not retrieved:
        docsearch = Searcher(anserini_path)
        searcher = docsearch.build_searcher(k1=3.44, b=0.87, index_path=index_path, rm3=True)  # TODO: what to do about the parameters?
        docsearch.search_document(searcher, None, qid2text, output_fn, collection, K=1000)
    else:
        # In the corpus tsv, each docid occurs at offset docoffset[docid]
        docoffset = {}
        with gzip.open(os.path.join(data_path, 'collection', 'msmarco-docs-lookup.tsv.gz'), 'rt') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [docid, _, offset] in tsvreader:
                docoffset[docid] = int(offset)

        qidx = 0
        didx = 1
        old_qid = -1
        with open(os.path.join(data_path, 'runs', 'run.msmarco-doc-expanded-original.bm25-default+rm3.topics.msmarco-doc.test.txt'), 'r') as top100f, \
            open(output_fn, 'w', encoding='utf-8') as out, open(os.path.join(data_path, 'collection', 'msmarco-docs.tsv'), encoding='utf8') as f:
            for line in top100f:
                qid, _, docid, rank, score, _ = line.strip().split(' ')
                if qid != old_qid:
                    qidx += 1
                    old_qid = qid
                text = qid2text[qid]
                content = getcontent(docid, f)
                clean_content = clean_html(content, collection=collection)
                tokenized_content = tokenizer.tokenize(clean_content)
                sentid = 0
                for sent in tokenized_content:
                    # Split sentence if it's longer than BERT's maximum input length
                    if len(sent.strip().split()) > MAX_INPUT_LENGTH:
                        seq_list = chunk_sent(sent, MAX_INPUT_LENGTH)
                        for seq in seq_list:
                            sentno = docid + '_' + str(sentid)
                            out.write('0\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(round(float(score), 11), text, seq, qid, sentno, qidx, didx - 1))
                            out.flush()
                            sentid += 1
                            didx += 1
                    else:
                        sentno = docid + '_' + str(sentid)
                        out.write('0\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(round(float(score), 11), text, sent, qid, sentno, qidx, didx - 1))
                        out.flush()
                        sentid += 1
                        didx += 1
