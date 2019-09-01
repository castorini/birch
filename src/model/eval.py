import subprocess
import shlex


def evaluate(trec_eval_path, predictions_file, qrels_file):
    cmd = trec_eval_path + ' {judgement} {output} -m map -m P.20 -m ndcg_cut.20'.format(judgement=qrels_file, output=predictions_file)
    pargs = shlex.split(cmd)
    print('Running {}'.format(cmd))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    lines = pout.split(b'\n')

    map = float(lines[0].strip().split()[-1])
    p20 = float(lines[1].strip().split()[-1])
    ndcg20 = float(lines[2].strip().split()[-1])

    return map, p20, ndcg20
