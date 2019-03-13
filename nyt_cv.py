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
    html = unicode(html, errors='ignore')
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
    cleaned = re.sub(r"\n", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
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

def get_qid2query(ftopic):
    qid2query = {}
    f = open(ftopic)
    query_tag = "title"
    empty = False
    for l in f:
        if empty == True:
            qid2query[qid] = l.replace("\n", "").strip()
            empty = False
        ind = l.find("Number: ")
        if ind >= 0:
            qid = l[ind+8:-1]
            qid = str(int(qid))
        ind = l.find("<{}>".format(query_tag))
        if ind >= 0:
            query = l[ind+8:-1].strip()
            if len(query) == 0:
                empty = True
            else:
                qid2query[qid] = query
    return qid2query

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

def search_document(searcher, prediction_fn, qid2text,
        output_fn, qid2reldocids, qidx, didx, K=1000):
    # f = open(prediction_fn, "w")
    out = open(output_fn, "w")
    method = "rm3"
    for qid in qid2text:
        a = qid2text[qid]
        print qid, a
        hits = searcher.search(JString(a), K)
        for i in range(len(hits)):
            sim = hits[i].score
            docno = hits[i].docid
            label = 1 if qid in qid2reldocids and docno in qid2reldocids[qid] else 0
            b = hits[i].content
            # b = "".join(filter(lambda x: x in printable, b))
            clean_b = clean_html(b)
            sent_id = 0
            # f.write("{} 0 {} 0 {} {}\n".format(qid, docno, sim, method))
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
    # f.close()
    out.close()
    return qidx, didx

def get_qid2reldocids(fqrel):
    f = open(fqrel)
    qid2reldocids = {}
    for l in f:
        qid, _, docid, score = l.replace("\n", "").strip().split()
        if score != "0":
            if qid not in qid2reldocids:
                qid2reldocids[qid] = set()
            qid2reldocids[qid].add(docid)
    return qid2reldocids




if __name__ == '__main__':
    fqrel = "../src/main/resources/topics-and-qrels/qrels.core17.txt"
    qid2reldocids = get_qid2reldocids(fqrel)
    ftopic = "../src/main/resources/topics-and-qrels/topics.core17.txt"
    qid2text = get_qid2query(ftopic)
    prediction_fn = "predict_nyt_rm3_cv.txt"
    output_fn = "nyt_bm25_rm3_cv.txt"
    index_path="/tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs"
    folder_idx = 1
    qidx, didx = 1, 1

    searcher = build_searcher(k1=0.9, b=0.4,
        index_path=index_path, rm3=True)
    # searcher = build_searcher()
    search_document(searcher, prediction_fn, qid2text, output_fn, 
        qid2reldocids, qidx, didx, 1000)

