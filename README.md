# 대화 맥락 기반 부적절 발언 탐지 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '대화 맥락 기반 부적절 발언 탐지'에 대한 베이스라인 모델의 추론을 재현하기 위한 코드를 포함하고 있습니다.  

추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

## 리포지토리 구조 (Repository Structure)
```
# 추론에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
└── test.py

# 추론에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
    {
        "id": "nikluge-2025-au-dev-00001",
        "input":
        {
            "document_id": "nikluge-2025-au-dev-00001",
            "utterance":
            [
                {
                    "id": "nikluge-2025-au-dev-00001-001",
                    "speaker_id": "P1",
                    "form": "중고 가구를 팔면서 속을 안 살피는 사람이 어딨어, 무겁고 소리도 났을 텐데."
                },
                {
                    "id": "nikluge-2025-au-dev-00001-002",
                    "speaker_id": "P2",
                    "form": "자작나무 타는 냄새가 여기까지 진동한다.ㅋ"
                }
            ]
        },
        "output":
        [
            {
                "id": "nikluge-2025-au-dev-00001-001",
                "label": "appropriate"
            },
            {
                "id": "nikluge-2025-au-dev-00001-002",
                "label": "inappropriate"
            }
        ]
    }
]
```

## 실행 방법 (How to Run)
### 추론 (Inference)
```
(실제 코드는 25년 7월 중순에 업데이트 예정)
```


## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)
