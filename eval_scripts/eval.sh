experiment=$1
collection=$2
anserini_path=$3
data_path=$4

echo "Experiment: ${experiment}"

if [[ ${experiment} == "baseline" ]] ; then
    echo "BM25+RM3:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${data_path}/qrels/qrels.${collection}.txt" "runs/run.${collection}.bm25+rm3.txt"
else
    echo "1S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${data_path}/qrels/qrels.${collection}.txt"  "runs/run.${experiment}.cv.a"

    echo "2S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${data_path}/qrels/qrels.${collection}.txt"  "runs/run.${experiment}.cv.ab"

    echo "3S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${data_path}/qrels/qrels.${collection}.txt"  "runs/run.${experiment}.cv.abc"
fi
