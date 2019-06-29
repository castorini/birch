import sys
import re
import nltk
import ssl
from importlib import reload

reload(sys)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

MAX_INPUT_LENGTH = 512


def get_query(ftopic, collection):
    qid2query = {}
    empty = False
    qid = -1
    with open(ftopic) as f:
        for line in f:
            if empty is True and int(qid) >= 0:
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
                query = line[ind + len(tag) + 3:-1].strip()

                if len(query) == 0:
                    empty = True
                else:
                    qid2query[qid] = query.lower()

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
    html = str(html)
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
