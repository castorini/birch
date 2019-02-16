from collections import defaultdict
import numpy as np
import operator
import sys
import re
import random
import os.path


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
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"\n", " ", cleaned)
    return cleaned.strip()


def load_HARD_docs(docF):
    doc_list = []
    with open(docF) as dF:
        for line in dF:
            doc = line.strip()
            doc_list.append(doc)
    return doc_list 

def load_HARD_topics(topicF):
    topic_dict = {}
    with open(topicF) as tF:
        for line in tF:
            topic, desc = line.strip().split('\t') 
            topic_dict[topic] = desc
    return topic_dict

def load_HARD_sent_qrels(topic_dict,path, doc_list):
    rel_dict = defaultdict(list)
    with open('qrels.sent') as pF:
        for line in pF:
            topic, _, sent = line.strip().split()
            doc = sent.rsplit('.', 1)[0]
            number = int(sent.rsplit('.', 1)[1])
            rel_dict[doc].append(number)
    
    neg_dict  = defaultdict(list)
    for doc in rel_dict:
        max_sent = max(rel_dict[doc])
        for i in range(0, max_sent):
            if i not in rel_dict[doc]:
                neg_dict[doc].append(i)


    with open('qrels.sent') as pF:
        for line in pF:
            topic, _, sent = line.strip().split()
            f = open(path+sent, "r")
            content = f.read()
            print '1'+'\t'+topic_dict[topic]+'\t'+clean_html(content) 
            
            doc = sent.rsplit('.', 1)[0]
            number = int(sent.rsplit('.', 1)[1])
            if doc in neg_dict:
                neg_list = neg_dict[doc]
                random.shuffle(neg_list)
                for neg_sent_num in neg_list:
                    neg_file = doc+'.'+str(neg_sent_num)
                    if os.path.isfile(path+neg_file):
                        neg_dict[topic].append(neg_file)
                        neg_f = open(path+neg_file, "r")
                        neg_content = neg_f.read()
                        print '0'+'\t'+topic_dict[topic]+'\t'+ \
                            ' '.join(clean_html(neg_content).split()[:50])
                        break


def main():
    doc_list = load_HARD_docs('hard.doc')
    topic_dict = load_HARD_topics('topics')
    load_HARD_sent_qrels(topic_dict, '/home/gvcormac/HARD/', doc_list)

if __name__ == "__main__":
    main()