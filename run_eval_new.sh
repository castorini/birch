    judgement=$2
output=$1

./etc/trec_eval.9.0/trec_eval ${judgement} ${output} -m map -m recip_rank -m P.20,30 -m ndcg_cut.20
exit 0 
