# DP로 MIA 방어하기 실습

**물리학전공 / 202540487 / 송영빈**

**대규모 언어 모델 보안 및 개인 정보 보호 [실습]**

## 실험 개요

### 목표
- Membership Inference Attack (MIA)에 대한 취약성 확인
- Differential Privacy (DP-SGD)를 적용하여 MIA 방어 효과 측정
- Privacy-Utility Trade-off 분석

### 환경
- **데이터셋**: IMDb 영화 리뷰 데이터셋 (50,000개)
- **모델**: DistilBERT (이진 분류)
- **DP 적용**: Opacus 패키지 활용

## 데이터셋 분할

```
IMDb 학습용 (25k)
├── A1: 타겟 모델 학습용 (멤버, 12.5k)
└── A2: MIA 평가용 멤버 (12.5k)

IMDb 테스트용 (25k)
├── B1: 새도우 모델 학습용 (6.25k)
├── B2: 새도우 모델 검증용 (6.25k)
└── B3: MIA 평가용 비멤버 (12.5k)

최종 MIA 평가: A2 (멤버) + B3 (비멤버)
```

## 실행 방법

### Linux 서버에서 실행

```bash
# 실행 권한 부여
chmod +x run.sh

# 실행
./run.sh
```

### 직접 실행

```bash
# 가상환경 생성 (선택)
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 실행
python3 dp_mia_defense.py
```

## 예상 결과

```
=== 최종 요약 ===
[비보호 타겟] 분류 - A2(holdout): 0.8270, B3(nonmember): 0.8415
[DP 타겟]     분류 - A2(holdout): 0.6335, B3(nonmember): 0.6310

[비보호 타겟] Shadow+MLP MIA  - 정확도: 0.4960, AUC: 0.4962
[DP 타겟]     Shadow+MLP MIA  - 정확도: 0.5070, AUC: 0.5112

[비보호 타겟] SimpleConf MIA  - AUC: 0.6782
[DP 타겟]     SimpleConf MIA  - AUC: 0.5012

[DP 타겟]     epsilon(): 7.99, delta=1e-5 (설정값)
```

## 핵심 분석

| 지표 | 비보호 | DP 적용 | 변화 |
|------|--------|---------|------|
| 분류 정확도 | ~83% | ~63% | ↓ 20% |
| SimpleConf MIA AUC | ~0.68 | ~0.50 | ↓ (랜덤 수준) |

- **DP-SGD 적용 후 MIA AUC가 0.5에 가까워짐** → 랜덤 추측 수준으로 프라이버시 향상
- **단, 모델 정확도 하락** → Privacy-Utility Trade-off

## 파일 구조

```
dp_mia_lab/
├── dp_mia_defense.py   # 메인 실습 코드
├── requirements.txt    # 필요 패키지
├── run.sh              # 리눅스 실행 스크립트
└── README.md           # 설명 문서
```

## DP-SGD 하이퍼파라미터

```python
noise_multiplier = 1.0   # 노이즈 수준 (높을수록 강한 프라이버시)
max_grad_norm = 1.0      # 그래디언트 클리핑 (작을수록 강한 클리핑)
delta = 1e-5             # Privacy budget delta
```

## 참고

- [Opacus Documentation](https://opacus.ai/)
- [Membership Inference Attacks Against Machine Learning Models (Shokri et al.)](https://arxiv.org/abs/1610.05820)


