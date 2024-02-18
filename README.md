# SynDepAT
SynDepAT learns to transfer syntactic dependency knowledge across domains. Additionally, Introducing a Span-Relation (SpanRel) approach based on universal dependency transfer, which can be easily extended to a wide range of natural language analysis tasks. 

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

# Performance

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| ELECTRA-base-Syn-LSTM-CRF  | Chinese-XXX-NER  | chinese-electra-180g-base-discriminator |-  |-  |-  |
| ELECTRA-base-Syn-LSTM-Span | chinese-XXX-NER  | chinese-electra-180g-base-discriminator |- |-  | - |
| ELECTRA-base-Syn-LSTM-Span | chinese-XXX-NER  | chinese-electra-180g-base-discriminator |- |-  | - |
| ELECTRA-base-Syn-LSTM-Span | chinese-XXX-NER  | chinese-electra-180g-base-discriminator |- |-  | - |
| ELECTRA-base-Syn-LSTM-Span | chinese-XXX-NER  | chinese-electra-180g-base-discriminator |- |-  | - |

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| RoBERTa-base-Syn-LSTM-CRF  | English-Conll2003-NER  | RoBERTa-base |-  |-  |-  |
| RoBERTa-base-Syn-LSTM-Span | English-XXX-NER  | RoBERTa-base |- |-  | - |
| RoBERTa-base-Syn-LSTM-Span | chinese-XXX-NER  | RoBERTa-base |- |-  | - |
| RoBERTa-base-Syn-LSTM-Span | chinese-XXX-NER  | RoBERTa-base |- |-  | - |
| RoBERTa-base-Syn-LSTM-Span | chinese-XXX-NER  | RoBERTa-base |- |-  | - |
