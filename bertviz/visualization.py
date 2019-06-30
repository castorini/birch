# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes made by Jesse Vig on 12/12/18:
# - Adapted to BERT model
#

import torch

import os
import sys

# Switch to local pytorch_pretrained_bert for visualization
sys.path.insert(0, os.path.join(os.getcwd(), 'birch', 'bertviz'))
import bertviz.pytorch_pretrained_bert as pytorch_pretrained_bert


class AttentionVisualizer:

    def __init__(self, model_path):
        state_dict = torch.load(model_path)
        self.model, self.tokenizer = state_dict['model'], state_dict['tokenizer']
        self.model.output_attentions = True
        self.model = self.model.cuda()
        self.model.eval()
        self.tokenizer.never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

    def get_viz_data(self, sentence_a, sentence_b):
        tokens_tensor, token_type_tensor, tokens_a, tokens_b = self._get_inputs(sentence_a, sentence_b)
        tokens_tensor = tokens_tensor.cuda()
        token_type_tensor = token_type_tensor.cuda()
        attn = self._get_attention(tokens_tensor, token_type_tensor)
        return tokens_a, tokens_b, attn

    def _get_inputs(self, sentence_a, sentence_b):
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)
        tokens_a_delim = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b_delim = tokens_b + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_a_delim + tokens_b_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim) + [1] * len(tokens_b_delim)])
        return tokens_tensor, token_type_tensor, tokens_a_delim, tokens_b_delim

    def _get_attention(self, tokens_tensor, token_type_tensor):
        _, _, attn_data_list = self.model(tokens_tensor, token_type_ids=token_type_tensor)
        attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
        attn_tensor = attn_tensor.cpu()
        return attn_tensor.data.numpy()
