import json



with open('../trec_dl/msmarco-test2019-queries.tsv', 'r') as in_file:
    big_lst = []
    new_lst = []
    for line_no, line in enumerate(in_file):
        qid, _ = line.strip().split('\t')
        if int(line_no) % 40 == 0 and int(line_no) > 0:
            assert len(new_lst) == 40
            big_lst.append(new_lst)
            new_lst = []
        new_lst.append(qid)
    big_lst.append(new_lst)

with open('../trec_dl/msmarco-test2019-queries-folds.json', 'w') as out_file:
    assert len(big_lst) == 5
    out_file.write(json.dumps(big_lst))
