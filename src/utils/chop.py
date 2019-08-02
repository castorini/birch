import sys

window = int(sys.argv[1])
stride = int(sys.argv[2])

with open('data/datasets/robust04_5cv.csv', 'r') as in_file, open('data/datasets/robust04_5cv_t_w{}_s{}.csv'.format(window, stride), 'w') as out_file:
    counter = 0
    for in_line in in_file:
        label, sim, a, b, qno, docno, qid, docid = in_line.strip().split('\t')
        start = 0
        tokens = b.strip().split(' ')
        while start + window <= len(tokens):
            out_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, a, ' '.join(tokens[start:(start + window)]), qno, docno.split('_')[0] + '_' + str(counter), qid, counter))
            start += stride
            counter += 1
        out_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, sim, a, ' '.join(tokens[start:]), qno, docno.split('_')[0] + '_' + str(counter), qid, counter))
        counter += 1
