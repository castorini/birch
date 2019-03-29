def get_query(ftopic):
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
                qid = str(int(line[ind + len(tag):-1]))
            # Get topic title
            tag = 'title'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
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
            if score != "0":
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
