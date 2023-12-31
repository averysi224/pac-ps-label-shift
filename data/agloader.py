'''
Prepare the AG News dataset as tokenized torch
ids tensor and attention mask
'''

import torch
import transformers 
from datasets import load_dataset

class DataTensorLoader():
    def __init__(self, arch):

        allowed_arch = ['electra', 'bert', 'roberta']
        if arch not in allowed_arch:
            raise Exception('Invalid architecture, only allowed: electra, bert, roberta')
        self.arch = arch
        self.dataset = load_dataset('ag_news')
    
    def _get_data(self, data):
        texts = data['text']
        labels = data['label']

        # Tokenize and prep input tensors
        if self.arch == 'electra':
            tokenizer = transformers.ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']
        labels = torch.LongTensor(labels)

        return ids, mask, labels

    def get_train(self):
        return self._get_data(self.dataset['train'])

    def get_test(self):
        return self._get_data(self.dataset['test'])
