q2d = {}
with open('missing2.txt', 'r') as in_file, open('../trec_dl/msmarco_top1000_test.tsv', 'r') as original_file, open('missing_mbmsmarco_original.tsv', 'w') as out_file:
    for line in in_file:
        query, docid = line.strip().split(' ')
        if query not in q2d:
            q2d[query] = [docid]
        else:
            q2d[query].append(docid)
    for line in original_file:
        label, score, query, doc, qid, sid, qno, dno = line.strip().split('\t')
        did = sid.split('_')[0]
        if qid in q2d and did in q2d[qid]:
            out_file.write(line)
