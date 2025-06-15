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
                    "form": "이성 부부는 애 키우는데 돈 쓰는데 동성 부부는 애를 못 낳으니까 쓸 돈이 많은 건가?"
                },
                {
                    "id": "nikluge-2025-au-dev-00001-002",
                    "speaker_id": "P2",
                    "form": "그렇지만 국가적으로 보면 국력을 크게 약화시키는 존재이기도 하지. 자손을 안 가지니까 생산력 점점 갉아먹잖아."
                },
                {
                    "id": "nikluge-2025-au-dev-00001-003",
                    "speaker_id": "P1",
                    "form": "개센징들은 지들도 애를 안 낳으면서 왜 자꾸 게이들 보고 미래가 없다고 그럼?"
                },
                {
                    "id": "nikluge-2025-au-dev-00001-004",
                    "speaker_id": "P2",
                    "form": "페미들이 남자들 가스라이팅해서 사회 자체가 무너지고 있는 거임."
                }
            ]
        },
        "output":
        [
            {
                "id": "nikluge-2025-au-dev-00001-001",
                "label": "inappropriate"
            },
            {
                "id": "nikluge-2025-au-dev-00001-002",
                "label": "inappropriate"
            },
            {
                "id": "nikluge-2025-au-dev-00001-003",
                "label": "inappropriate"
            },
            {
                "id": "nikluge-2025-au-dev-00001-004",
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
