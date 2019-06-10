from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert import BertTokenizer
import torch

bert_model = 'bert-large-uncased'
model = BertForMaskedLM.from_pretrained(bert_model)
tokenizer = BertTokenizer.from_pretrained(bert_model)

question = 'who invented the telephone'  # "the telephone was invented by whom"
tokenized_question = tokenizer.tokenize(question)

masked_index = 0
tokenized_question[masked_index] = '[MASK]'
question_ids = tokenizer.convert_tokens_to_ids(tokenized_question)
combined_ids = question_ids
segments_ids = [0] * len(question_ids)

tokens_tensor = torch.tensor([combined_ids])
segments_tensor = torch.tensor([segments_ids])

model.eval()
predictions = model(tokens_tensor, segments_tensor)  # 1 x len(combined_ids) x vocab size
predicted_index = torch.topk(predictions[0, masked_index], 20)[1].tolist()
print(predicted_index)
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(predicted_token)