from transformers import ElectraModel
import torch.nn as nn

class ClassificationHeadElectra(nn.Module):
    '''Head for classifying sequence embeddings'''
    def __init__(self, hidden_size, classes, dropout=0.5):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, classes)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.layer(x)
        m = nn.GELU()
        x = m(x) # gelu used by electra authors
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraSequenceClassifier(nn.Module):
    '''
    Sentence Level classification using Electra for encoding sentence
    '''
    def __init__(self, hidden_size=768, classes=4):
        super().__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-base-discriminator')
        self.classifier = ClassificationHeadElectra(hidden_size, classes)

    def forward(self, input_ids, attention_mask):
        '''
        input_ids = [N x L], containing sequence of ids of words after tokenization
        attention_mask = [N x L], mask for attention

        N = batch size
        L = maximum sentence length
        '''
        all_layers_hidden_states = self.electra(input_ids, attention_mask)
        final_layer = all_layers_hidden_states[0]
        logits = self.classifier(final_layer)
        return logits