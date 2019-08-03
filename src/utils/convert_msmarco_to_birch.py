import sys
from nltk.tokenize import sent_tokenize


def convert(in_path, out_path):
    with open(in_path, 'r') as in_file, open(out_path, 'w') as out_file:
        for in_line in in_file:
            qid, query, pos_did, pos_url, pos_title, pos_doc, \
            neg_did, neg_url, neg_title, neg_doc = in_line.strip().split('\t')

            # Split into sentences
            sents = sent_tokenize(pos_doc)
            for sent_id, sent in enumerate(sents):
                out_file.write('1\t{}\t{}\t{}\n'.format(query, sent, qid))

            sents = sent_tokenize(neg_doc)
            for sent_id, sent in enumerate(sents):
                out_file.write('1\t{}\t{}\t{}\n'.format(query, sent, qid))


triples_file = sys.argv[1]
birch_file = sys.argv[2]

convert(triples_file, birch_file)