import sys

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def main():
    dataset = sys.argv[1]
    stop_words = list(stopwords.words('english'))
    with open(dataset, 'r') as in_file, \
        open(dataset + '_pruned', 'w') as out_file:
        for line in in_file:
            label, score, query, sent, qid, sid, qno, sno = line.strip().split('\t')
            lower_query = [tok.lower() for tok in filter(lambda t: t not in stop_words, nltk.word_tokenize(query))]
            lower_sent = [tok.lower() for tok in nltk.word_tokenize(sent)]
            contains = False
            for q in lower_query:
                if q in lower_sent:
                   contains = True
            if contains:
                out_file.write(line)


if __name__ == '__main__':
    main()
