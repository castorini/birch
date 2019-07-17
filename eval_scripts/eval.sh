experiment=$1
anserini_path=$2
qrels_file=$3

echo "Experiment: ${experiment}"

if [[ ${experiment} == *"bm25+rm3"* ]] ; then
    echo "BM25+RM3:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.txt"
else
    echo "1S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.a"

    echo "2S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.ab"

    echo "3S:"
    ${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 -m ndcg_cut.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.abc"
fi
