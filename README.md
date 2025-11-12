# Capdol_Unsmile

```c
# 토크나이저 및 모델 불러오기 위한 import 작업
from transformers import BertForSequenceClassification, BertTokenizer
import torch

model_name = "Huffon/klue-roberta-base-nli"
model_name = 'klue/roberta-base'
model_name = 'klue/roberta-large'
model_name = 'beomi/kcbert-base'


# 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(data_train['labels'][0]))


# 모델 학습 준비
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
```
