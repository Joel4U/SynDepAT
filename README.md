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

 
| NER Model  | Dataset | Pretrained Model |P | R | F1 | Parameters |备注 |
| ----------------------- | ------------- |-------------|--------- |--------- |--------- |------------- |------------- |
| SynDepAT （w/o dep）-crf | conll03| bert-base-cased |91.97 |92.10  |92.67  | bert-lr: 2e-5  other-lr: 1e-3  (lstm)enc_layers: 1, (lstm)shared_enc_nlayers: 3|基本是突变，正常F1就在91.96左右
| SynDepAT （w/o dep）-crf|  ontonotes  | bert-base-cased |89.59 |89.62  | 89.61 | 同上 |
| SynDepAT （w/o dep）-crf|  chinese  | bert-base-multilan-cased |77.62 | 80.98  | 79.27 | 同上 |
| SynDepAT （w/o dep）-crf|  resume | bert-base-multilan-cased |95.73 |6.38  | 96.06 | - |
| SynDepAT （w/o dep）-crf|  catalan  | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-crf|  spanish | bert-base-multilan-cased |- |-  | - | - |
| ----------------------- | ------------- |-------------|--------- |--------- |--------- |------------- |
| SynDepAT （w/o dep）-biaf | conll03| bert-base-cased |-  |-  |-  | - |
| SynDepAT （w/o dep）-biaf|  ontonotes  | bert-base-cased |- |-  | - | - |
| SynDepAT （w/o dep）-biaf|  chinese  | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-biaf|  resume | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-biaf|  catalan  | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-biaf|  spanish | bert-base-multilan-cased |- |-  | - | - |
| ----------------------- | ------------- |-------------|--------- |--------- |--------- |------------- |
| SynDepAT （w/o dep）-span | conll03| bert-base-cased |-  |-  |-  | - |
| SynDepAT （w/o dep）-span|  ontonotes  | bert-base-cased |- |-  | - | - |
| SynDepAT （w/o dep）-span|  chinese  | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-span|  resume | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-span|  catalan  | bert-base-multilan-cased |- |-  | - | - |
| SynDepAT （w/o dep）-span|  spanish | bert-base-multilan-cased |- |-  | - | - |
