# OntoNote 4.0
# train
#python main.py --mode train --batch_size 8 --eval_steps 1000 --data_name onto4ner.cn --data_path data --learning_rate 3e-5 --num_train_epochs 10 --pytorch_dump_path saved/ontonote/saved.model --data_format ontonote --chinese --output_path predict.ontonote --model_type BertForTokenClassification --num_labels 17
# test
python main.py --mode test --batch_size 2 --eval_steps 1000 --data_name onto4ner.cn --data_path data --learning_rate 3e-5 --num_train_epochs 10 --pytorch_dump_path saved/ontonote/saved.model_3 --data_format ontonote --chinese --output_path predict.ontonote --model_type BertForTokenClassification --num_labels 17

# Resume
#python main.py --mode train --batch_size 8 --eval_steps 1000 --data_name ResumeNER --data_path data --learning_rate 3e-5 --num_train_epochs 10 --pytorch_dump_path saved/ResumeNER/saved.model --data_format ontonote --chinese --output_path predict.ResumeNER --model_type BertForTokenClassification --num_labels 28

# MSRA
#python main.py --mode train --batch_size 8 --eval_steps 1000 --data_name msra --data_path data --learning_rate 3e-5 --num_train_epochs 10 --pytorch_dump_path saved/msra/saved.model --data_format ontonote --chinese --output_path predict.msra --model_type BertForTokenClassification --num_labels 13
