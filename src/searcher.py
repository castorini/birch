import os
import json
from utils import parse_doc_from_index, clean_html, tokenizer, MAX_INPUT_LENGTH, chunk_sent

import jnius_config
import glob

class Searcher:
    def __init__(self, anserini_path):
        paths = glob.glob(os.path.join(anserini_path, 'target', 'anserini-*-fatjar.jar'))
        if not paths:
            raise Exception("No matching jar file for Anserini found in target")

        latest = max(paths, key=os.path.getctime)
        jnius_config.set_classpath(latest)

        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        self.JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.qidx = 1
        self.didx = 1

    def reset_idx(self):
        self.qidx = 1
        self.didx = 1

    def build_searcher(self, k1=0.9, b=0.4, fb_terms=10, fb_docs=10, original_query_weight=0.5,
        index_path="index/lucene-index.robust04.pos+docvectors+rawdocs", rm3=False):
        searcher = self.JSearcher(self.JString(index_path))
        searcher.setBM25Similarity(k1, b)
        if not rm3:
            searcher.setDefaultReranker()
        else:
            searcher.setRM3Reranker(fb_terms, fb_docs, original_query_weight, False)
        return searcher

    def search_document(self, searcher, qid2docid, qid2text, output_fn, collection='robust04', K=1000, topics=None, cv_fold=None):
        output_dir = os.path.dirname(output_fn)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_fn, 'w', encoding="utf-8") as out:
            if 'core' in collection:
                # Robust04 provides CV topics
                topics = qid2text
            for qid in topics:
                text = qid2text[qid]
                hits = searcher.search(self.JString(text), K)
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
                    if collection == 'robust04':
                        content = parse_doc_from_index(content)
                    clean_content = clean_html(content, collection=collection)
                    tokenized_content = tokenizer.tokenize(clean_content)
                    sentid = 0
                    for sent in tokenized_content:
                        # Split sentence if it's longer than BERT's maximum input length
                        if len(sent.strip().split()) > MAX_INPUT_LENGTH:
                            seq_list = chunk_sent(sent, MAX_INPUT_LENGTH)
                            for seq in seq_list:
                                sentno = docno + '_' + str(sentid)
                                if cv_fold == '5': 
                                    out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, round(float(sim), 11), text, seq, qid, sentno, qid, self.didx-1))
                                else:
                                    out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, round(float(sim), 16), text, seq, qid, sentno, self.qidx, self.didx))
                                out.flush()
                                sentid += 1
                                self.didx += 1
                        else:
                            sentno = docno + '_' + str(sentid)
                            if cv_fold == '5': 
                                out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, round(float(sim), 11), text, sent, qid, sentno, qid, self.didx-1))
                            else:
                                out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, round(float(sim), 16), text, sent, qid, sentno, self.qidx, self.didx))
                            out.flush()
                            sentid += 1
                            self.didx += 1
                self.qidx += 1
