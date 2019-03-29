import string
printable = set(string.printable)
printable.remove("\n")
printable.remove("\t")
printable.remove("\r")
import os
import shlex
import subprocess
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import json

from args import get_args
from robust04_utils import *

from searcher import *
import searcher

import nltk
nltk.download('punkt')

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.
    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"\t", " ", cleaned)
    return cleaned.strip()

def chunk_sent(sent, max_len):
    chunked_sents = []
    idx = 0
    words = sent.strip().split()
    size = int(len(words)/max_len)
    for i in range(0, size):
        seq = words[i * max_len: (i+1) * max_len]
        chunked_sents.append(" ".join(seq))
    return chunked_sents

def parse_doc_from_index(content):
    ls = content.split("\n")
    see_text = False
    doc = ""
    for l in ls:
        l = l.replace("\n", "").strip()
        if "<TEXT>" in l:
            see_text = True
        elif "</TEXT>" in l:
            break
        elif see_text:
            if l == "<P>" or l == "</P>":
                continue
            doc += l + " "
    return doc.strip()

def search_document(topics, searcher, qid2text, qid2desc,
                    output_fn, qid2docid, qidx, didx, K=1000):
    with open(output_fn, 'w') as out:
        # TODO: refactor
        method = "rm3"
        for qid in topics:
            a = qid2text[qid]
            desc = qid2desc[qid]
            hits = searcher.search(JString(a), K)
            for i in range(len(hits)):
                sim = hits[i].score
                docno = hits[i].docid
                label = 1 if qid in qid2docid and docno in qid2docid[qid] else 0
                b = parse_doc_from_index(hits[i].content)
                # b = "".join(filter(lambda x: x in printable, b))
                clean_b = clean_html(b)
                sent_id = 0
                for sentence in tokenizer.tokenize(clean_b):
                    if len(sentence.strip().split()) > 512:
                        seq_list = chunk_sent(sentence, 512)
                        for seq in seq_list:
                            sentno = docno + '_'+ str(sent_id)
                            # f.write("{} 0 {} 0 {} {}\n".format(qid, sentno, sim, method))
                            out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(label, sim, a, seq, qid, sentno, qidx, didx))
                            out.flush()
                            sent_id += 1
                            didx += 1
                    else:
                        sentno = docno + '_'+ str(sent_id)
                        # f.write("{} 0 {} 0 {} {}\n".format(qid, sentno, sim, method))
                        out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(label,\
                            sim, a, sentence, qid, sentno, qidx, didx))
                        out.flush()
                        sent_id += 1
                        didx += 1
            qidx += 1
        out.close()
        return qidx, didx

def cal_score(fn_qrels="../Anserini/src/main/resources/topics-and-qrels/qrels.robust2004.txt", prediction="score.txt"):
    cmd = "/bin/sh run_eval_new.sh {} {}".format(prediction, fn_qrels)
    pargs = shlex.split(cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()
    print("running {}".format(cmd))
    if sys.version_info[0] < 3:
        lines = pout.split('\n')
    else:
        lines = pout.split(b'\n')
    Map = float(lines[0].strip().split()[-1])
    Mrr = float(lines[1].strip().split()[-1])
    P20 = float(lines[2].strip().split()[-1])
    P30 = float(lines[3].strip().split()[-1])
    NDCG20 = float(lines[4].strip().split()[-1])
    print(Map)
    print(Mrr)
    print(P30)
    print(P20)
    print(NDCG20)
    return Map, Mrr, P30, P20, NDCG20


if __name__ == '__main__':
    args = get_args()
    target_path = args.target_path
    data_path = args.data_path
    index_path = args.index_path
    folds_path = args.folds_path
    cv_folds = args.cv_folds
    output_fn = args.output_path
    prediction_fn = args.prediction_path

    fqrel = os.path.join(data_path, 'topics-and-qrels', 'qrels.robust2004.txt')
    ftopic = os.path.join(data_path, 'topics-and-qrels', 'topics.robust04.301-450.601-700.txt')

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(ftopic)
    qid2desc = get_desc('topics.desc')

    if cv_folds == 5:
        with open(os.path.join(folds_path, 'robust04-paper2-folds.json')) as f:
            folds = json.load(f)
        with open(os.path.join(folds_path, 'robust04-paper2-folds-map-params.json')) as f:
            params = json.load(f)
    elif cv_folds == 2:
        with open(os.path.join(folds_path, 'robust04-paper1-folds.json')) as f:
            folds = json.load(f)
        # TODO: params?
        params = [  "0.9 0.5 50 17 0.20",
                    "0.9 0.5 26 8 0.30"]

    folder_idx = 1
    query_idx, doc_idx = 1, 1  # TODO: ?
    for topics, param in zip(folds, params):
        # Extract each parameter
        k1, b, fb_terms, fb_docs, original_query_weight = map(float, param.strip().split())
        searcher = build_searcher(k1=k1, b=b, fb_terms=fb_terms, fb_docs=fb_docs,
                                  original_query_weight=original_query_weight,
                                  index_path=index_path, rm3=True)

        print(query_idx)

        query_idx, doc_idx = search_document(topics, searcher, qid2text, qid2desc,
                                            output_fn + str(folder_idx), qid2docid,
                                            query_idx, doc_idx, 1000)

        folder_idx += 1
        # cal_score(prediction=prediction_fn)
