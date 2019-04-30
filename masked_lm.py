from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
from pytorch_pretrained_bert import BertTokenizer
import torch

bert_model = 'bert-large-uncased'
model = BertForMaskedLM.from_pretrained(bert_model)
tokenizer = BertTokenizer.from_pretrained(bert_model)

question = "Who discovered relativity ?"
answer = "Einstein or Newton or Bohr"
tokenized_question = tokenizer.tokenize(question)
tokenized_answer = tokenizer.tokenize(answer)

masked_index = 0  # Who
tokenized_question[masked_index] = '[MASK]'
question_ids = tokenizer.convert_tokens_to_ids(tokenized_question)
answer_ids = tokenizer.convert_tokens_to_ids(tokenized_answer)
print(answer_ids[2])
combined_ids = question_ids + answer_ids
segments_ids = [0] * len(question_ids) + [1] * len(answer_ids)

tokens_tensor = torch.tensor([combined_ids])
segments_tensor = torch.tensor([segments_ids])

model.eval()
predictions = model(tokens_tensor, segments_tensor)  # 1 x len(combined_ids) x vocab size
predicted_index = torch.topk(predictions[0, masked_index], 300)[1].tolist()
print(predicted_index)
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(predicted_token)