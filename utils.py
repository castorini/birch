import sys
import re
import json
import nltk
from searcher import JString

reload(sys)
sys.setdefaultencoding('utf-8')

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

MAX_INPUT_LENGTH = 512


# Corpus functions
def get_query(ftopic, collection):
    qid2query = {}
    empty = False
    qid = -1
    with open(ftopic) as f:
        for line in f:
            if empty is True and qid >= 0:
                qid2query[qid] = line.replace('\n', '').strip()
                empty = False
            # Get topic number
            tag = 'Number: '
            ind = line.find(tag)
            if ind >= 0:
                # Remove </num> from core18
                end_ind = -7 if collection == 'core18' else -1
                qid = str(int(line[ind + len(tag):end_ind]))
            # Get topic title
            tag = 'title'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
                if 'core' in collection:
                    # Remove leading tag
                    query = line[ind + len(tag) + 3:-1].strip()
                else:
                    query = line[ind + len(tag):-1].strip()
                if len(query) == 0:
                    empty = True
                else:
                    qid2query[qid] = query

    return qid2query


def get_relevant_docids(fqrel):
    qid2docid = {}
    with open(fqrel) as f:
        for line in f:
            qid, _, docid, score = line.replace('\n', '').strip().split()
            if score != '0':
                if qid not in qid2docid:
                    qid2docid[qid] = set()
                qid2docid[qid].add(docid)
    return qid2docid


def get_desc(fdesc):
    qid2desc = {}
    with open(fdesc) as f:
        for line in f:
            qid, desc = line.strip().split('\t')
            qid2desc[qid] = desc
    return qid2desc


# Text processing functions
def chunk_sent(sent, max_len):
    chunked_sents = []
    words = sent.strip().split()
    size = int(len(words) / max_len)
    for i in range(0, size):
        seq = words[i * max_len: (i + 1) * max_len]
        chunked_sents.append(' '.join(seq))
    return chunked_sents


def clean_html(html, collection):
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
    if 'core' in collection:
        cleaned = re.sub(r"\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_doc_from_index(content):
    ls = content.split('\n')
    see_text = False
    doc = ''
    for l in ls:
        l = l.replace('\n', '').strip()
        if '<TEXT>' in l:
            see_text = True
        elif '</TEXT>' in l:
            break
        elif see_text:
            if l == '<P>' or l == '</P>':
                continue
            doc += l + ' '
    return doc.strip()


# Retrieval functions
def search_core(searcher, qid2docid, qid2text, output_fn, collection='core17', K=1000):
    qidx, didx = 1, 1
    with open(output_fn, 'w') as out:
        for qid in qid2text:
            text = qid2text[qid]
            # print(qid, text)
            hits = searcher.search(JString(text), K)
            for i in range(len(hits)):
                sim = hits[i].score
                docno = hits[i].docid
                label = 1 if qid in qid2docid and docno in qid2docid[qid] else 0
                content = hits[i].content
                if collection == 'core18':
                    content_json = json.loads(content)
                    content = ''
                    for each in content_json['contents']:
                        if each is not None and 'content' in each.keys():
                            content += '{}\n'.format(each['content'])
                clean_content = clean_html(content, collection=collection)
                tokenized_content = tokenizer.tokenize(clean_content)

                sentid = 0
                for sent in tokenized_content:
                    # Split sentence if it's longer than BERT's maximum input length
                    if len(sent.strip().split()) > MAX_INPUT_LENGTH:
                        seq_list = chunk_sent(sent, MAX_INPUT_LENGTH)
                        for seq in seq_list:
                            sentno = docno + '_' + str(sentid)
                            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, text, seq, qid, sentno, qidx, didx))
                            out.flush()
                            sentid += 1
                            didx += 1
                    else:
                        sentno = docno + '_' + str(sentid)
                        out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, text, sent, qid, sentno, qidx, didx))
                        out.flush()
                        sentid += 1
                        didx += 1
            qidx += 1
