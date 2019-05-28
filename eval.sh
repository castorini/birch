experiment=$1
qrels_file=$2

declare -a sents=("a" "b" "c")

for i in "${sents[@]}"
do
    ../Anserini/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m P.30 -m ndcg_cut.30 ${qrels_file} "run.${experiment}.cv.$i"
done