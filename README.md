# SynDepAT
SynDepAT learns to transfer syntactic dependency knowledge across domains. The dependency transfer source domain comes from the dependency parsing task, while the target task is NER. Of course, this method can be easily extended to a wide range of natural language analysis tasks.

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

# Performance

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| SynDepAT | ctb5 and ontonotes cn| bert-base-multilan-cased |-  |-  |-  |
| SynDepAT | ctb5 and resume  | bert-base-multilan-cased |- |-  | - |
| SynDepAT | ctb5 and weibo  | bert-base-multilan-cased |- |-  | - |
| SynDepAT | ctb5 and marsa  | bert-base-multilan-cased |- |-  | - |

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| SynDepAT | ptb and ontonotes en| bert-large-cased |-  |-  |-  |
| SynDepAT | ptb and conll03  | bert-large-cased |- |-  | - |
| SynDepAT | ptb and ace04  | bert-large-cased |- |-  | - |
| SynDepAT | ptb and ace05  | bert-large-cased |- |-  | - |
