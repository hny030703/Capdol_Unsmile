# Capdol_Unsmile
https://uiandwe.tistory.com/1395
해당 블로그를 참고하여 언어폭력을 예방하기 위한 모델을 제작하는 것이 목표이다.

```c
# 토크나이저 및 모델 불러오기 위한 import 작업
from transformers import BertForSequenceClassification, BertTokenizer
import torch

//model_name = "Huffon/klue-roberta-base-nli"
//model_name = 'klue/roberta-base'
//model_name = 'klue/roberta-large'
model_name = 'beomi/kcbert-base'


# 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(data_train['labels'][0]))


# 모델 학습 준비
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
```

#Bert 모델의 구조
Input layer : 입력 문장을 벡터로 변환
Embedding layer : 벡터를 transformer layer에 입력
Transformer Encoder : 문장 이해 및 추출
Output Layer : 라벨링 값 출력

*Bert 모델 내에 transformer 존재
정확히 말하면, Bert는 Transformer 인코더 부분만 사용한 모델이다.

```c
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at beomi/kcbert-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
BertForSequenceClassification(
  (bert): BertModel( // -> BertModel: embeddings, encoder, pooler layer로 구성되어있음
    (embeddings): BertEmbeddings( // -> 입력 토큰을 임베딩하는 부분 (단어, 위치, 토큰 타입 임베딩을 합침)
      (word_embeddings): Embedding(30000, 768, padding_idx=0)
      (position_embeddings): Embedding(300, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder( // -> Transformer 인코더 layer로 구성되어있음. 여러개의 BertLayer로 이루어져있음)
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler( // -> Bert의 출력을 사용해 시퀀스 레벨의 표현을 얻는 부분
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False) // -> 모델 내에서 사용되는 드롭아웃 layer. 과적합 방지용
  (classifier): Linear(in_features=768, out_features=10, bias=True) // -> 실제 분류를 수행하는 선형 분류기. out_features=10은 모델이 10개의 클래스를 분류할 수 있는 것을 나타냄
)
```

#키워드 설명
*과적합(overfitting): 모델이 훈련 데이터에는 너무 잘 맞지만, 새로운 테스트 데이터에서는 성능이 떨어지는 상태
*과적합을 막기 위해
- 데이터를 더 모은다.
- 정규화시킨다. (Dropout, L2 등으로 모델이 너무 복잡해지는 것을 방지)
- Early Stopping (검증 정확도가 떨어지기 시작하면 학습 중단, 더이상 정확도가 더 떨어지는 것을 막기 위해 학습을 멈추어야함)
- 데이터 증강 (Data Augmentation) 시킨다. (이미지 혹은 텍스트 데이터를 변형해서 데이터 양을 늘림)
- 모델을 단순화시킨다. (층 수나 파라미터 수를 줄임)

*클래스 개수: 10개 {'여성/가족'~'clean'}까지
('개인지칭'은 추가적인 정보이므로 멀티라벨 벡터에는 포함되지 않음)
