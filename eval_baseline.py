from collections import defaultdict
import numpy as np

def load_run(topK, topic, runF):
	with open(runF) as rF:
		topDoc = []
		for line in rF:
			t, _, sent, rank, score, run = line.split()
			if topic == t and int(rank) <= topK:
				doc = sent.split('_')[0]
				topDoc.append(doc)
	return topDoc

def load_nist_qrels():
    rel_dict = defaultdict(list) 
    all_dict = defaultdict(list)
    nonrel_dict = defaultdict(list)

    with open('qrels.robust2004.txt') as pF:
        for line in pF:
            topic, _, doc, label = line.split()
            all_dict[topic].append(doc)
            if int(label) > 0:
                rel_dict[topic].append(doc)
            else:
                nonrel_dict[topic].append(doc)
    return rel_dict, nonrel_dict, all_dict

def main():
    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    recall_list = []
    for topic in rel_dict:
    	topDoc = load_run(1000, topic, 'run.robust04.bm25.topics.robust04.301-450.601-700.txt')
    	ret = 0
    	rel = len(rel_dict[topic])
    	for doc in set(topDoc):
    		if doc in rel_dict[topic]:
    			ret+=1
    	# print topic, ret*1.0/rel
    	recall_list.append(ret*1.0/rel)
    print np.mean(recall_list)
if __name__ == "__main__":
    main()