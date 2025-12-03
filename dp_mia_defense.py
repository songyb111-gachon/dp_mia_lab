#!/usr/bin/env python3
"""
DP(Differential Privacy)로 MIA(Membership Inference Attack) 방어하기 실습

- IMDb 영화 리뷰 데이터셋
- DistilBERT 모델
- Opacus 패키지를 활용한 DP-SGD 적용

작성자: 송영빈 (202540487)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from datasets import load_dataset, concatenate_datasets
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 1. 환경 설정
# =====================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# =====================================
# 2. IMDb 데이터셋 불러오기
# =====================================
print("\n[Step 1] IMDb 데이터셋 로드 중...")
# IMDb 영화 리뷰 데이터셋 로드 (훈련(train)/테스트(test)로 구성)
imdb = load_dataset("imdb")
train_full = imdb["train"]   # 전체 훈련 데이터셋 (총 25,000개 예시)
test_full = imdb["test"]     # 전체 테스트 데이터셋 (총 25,000개 예시)

print(f"전체 훈련 샘플 수: {len(train_full)}, 전체 테스트 샘플 수: {len(test_full)}")

# =====================================
# 3. 데이터셋 분할
# =====================================
print("\n[Step 2] 데이터셋 분할 중...")

# A1 (멤버 학습), A2 (멤버 테스트용)
train_A1 = train_full.select(range(12500))              # 타겟 모델 학습용
train_A2 = train_full.select(range(12500, 25000))       # MIA 평가용 멤버

# B1, B2, B3: 테스트셋 분할
test_B1 = test_full.select(range(6250))                 # 새도우 모델 학습용
test_B2 = test_full.select(range(6250, 12500))          # 새도우 비멤버
test_B3 = test_full.select(range(12500, 25000))         # MIA 평가용 비멤버

# MIA 평가셋: 멤버는 A2, 비멤버는 B3
mia_eval_set = concatenate_datasets([train_A2, test_B3]).shuffle(seed=42)

print(f"A1 (타겟 학습): {len(train_A1)}, A2 (MIA 멤버): {len(train_A2)}")
print(f"B1 (새도우 학습): {len(test_B1)}, B2 (새도우 비멤버): {len(test_B2)}, B3 (MIA 비멤버): {len(test_B3)}")

# =====================================
# 4. 토큰화
# =====================================
print("\n[Step 3] 토큰화 중...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# 각 세트에 대해 토큰화 및 Tensor 포맷 적용
for ds in [train_A1, train_A2, test_B1, test_B2, test_B3, mia_eval_set]:
    ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 토큰화된 데이터셋 재생성 (map은 in-place가 아님)
train_A1_tok = train_A1.map(tokenize_batch, batched=True, remove_columns=["text"])
train_A1_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_A2_tok = train_A2.map(tokenize_batch, batched=True, remove_columns=["text"])
train_A2_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_B1_tok = test_B1.map(tokenize_batch, batched=True, remove_columns=["text"])
test_B1_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_B2_tok = test_B2.map(tokenize_batch, batched=True, remove_columns=["text"])
test_B2_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_B3_tok = test_B3.map(tokenize_batch, batched=True, remove_columns=["text"])
test_B3_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

mia_eval_tok = mia_eval_set.map(tokenize_batch, batched=True, remove_columns=["text"])
mia_eval_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print(f"  train_A1: {len(train_A1_tok)} 샘플")
print(f"  train_A2: {len(train_A2_tok)} 샘플")
print(f"  test_B1: {len(test_B1_tok)} 샘플")
print(f"  test_B2: {len(test_B2_tok)} 샘플")
print(f"  test_B3: {len(test_B3_tok)} 샘플")
print(f"  mia_eval_set: {len(mia_eval_tok)} 샘플")

# =====================================
# 5. Dataloader 준비
# =====================================
print("\n[Step 4] Dataloader 준비 중...")
# 배치 내에서 길이 다른 토큰 시퀀스를 자동으로 패딩
batch_size = 16  # 실습 환경에 따라 조정

# 타겟 모델 학습용 (A1)
train_loader_A1 = DataLoader(train_A1_tok, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
# MIA 평가 멤버용 (A2)
eval_loader_A2 = DataLoader(train_A2_tok, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
# MIA 평가 비멤버용 (B3)
eval_loader_B3 = DataLoader(test_B3_tok, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
# 새도우 모델 학습용 (B1)
train_loader_B1 = DataLoader(test_B1_tok, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
# 새도우 모델 비멤버 평가용 (B2)
eval_loader_B2 = DataLoader(test_B2_tok, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
# 최종 MIA 평가셋 (A2+B3)
mia_eval_loader = DataLoader(mia_eval_tok, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

print(f"배치 사이즈: {batch_size}")

# =====================================
# 6. DistilBERT 파인튜닝 (w/o DP) - 타겟 모델
# =====================================
print("\n" + "="*60)
print("[비보호 타겟 모델] DistilBERT 파인튜닝 (w/o DP)")
print("="*60)

# DistilBERT 이진 분류 모델 초기화 (사전학습 가중치 로드)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)  # 가능하면 GPU 활용

# 옵티마이저 및 손실 함수 정의
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 학습 루프
epochs = 1
model.train()
for epoch in range(epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch in tqdm(train_loader_A1, desc=f"Training Epoch {epoch+1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 학습 중 정확도 계산
        preds = outputs.logits.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    
    avg_loss = total_loss / len(train_loader_A1)
    train_acc = correct_train / total_train
    print(f"Epoch {epoch+1} 완료 - 평균 손실: {avg_loss:.4f}, 학습 정확도: {train_acc:.4f}")

# 타겟 모델 평가 (분류 정확도)
model.eval()
correct, total = 0, 0
first_batch = True
with torch.no_grad():
    for batch in eval_loader_A2:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        
        # 디버깅: 첫 배치의 라벨 확인
        if first_batch:
            print(f"[디버그] 평가 배치 키: {batch.keys()}")
            print(f"[디버그] 라벨 샘플: {labels[:5]}")
            first_batch = False
        
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_no_dp = correct / total
print(f"[비보호 타겟] 분류 정확도 - A2(holdout): {acc_target_no_dp:.4f}")

# B3 평가
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_B3:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_no_dp_b3 = correct / total
print(f"[비보호 타겟] 분류 정확도 - B3(nonmember): {acc_target_no_dp_b3:.4f}")

# =====================================
# 7. 새도우 모델 기반 MIA 구현 및 평가 (w/o DP 용)
# =====================================
print("\n" + "="*60)
print("[비보호 버전] 새도우 모델 기반 MIA 구현 및 평가")
print("="*60)

# 새도우 모델 초기화 (타겟 모델과 동일 구조)
shadow_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)
shadow_optimizer = optim.AdamW(shadow_model.parameters(), lr=2e-5)
shadow_model.train()

# 새도우 모델 학습 (데이터 B1 사용)
print("\n새도우 모델 학습 중...")
for epoch in range(1):  # 필요시 epoch 조절
    for batch in tqdm(train_loader_B1, desc="Shadow Model Training"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        outputs = shadow_model(**inputs, labels=labels)
        loss = outputs.loss
        shadow_optimizer.zero_grad()
        loss.backward()
        shadow_optimizer.step()

# Feature 수집 (새도우 모델의 멤버/비멤버 출력)
print("\nFeature 수집 중...")
shadow_model.eval()
member_features = []
for batch in train_loader_B1:
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    labels = batch["labels"].to(device)
    with torch.no_grad():
        outputs = shadow_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)  # 예측 확률분포
    member_features.extend(probs.cpu().tolist())  # list of [P(y=0), P(y=1)] for each sample
# 멤버 레이블 1 할당
member_labels = [1] * len(member_features)

# 비멤버 데이터 (B2)에 대해서 동일하게 수행
nonmember_features = []
shadow_model.eval()
for batch in eval_loader_B2:
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        outputs = shadow_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    nonmember_features.extend(probs.cpu().tolist())
nonmember_labels = [0] * len(nonmember_features)

print(f"멤버 특징 개수: {len(member_features)}, 비멤버 특징 개수: {len(nonmember_features)}")
# 두 리스트 모두 약 6250개여야 함.

# 공격 모델 학습
print("\n공격 모델 학습 중...")
# 학습 데이터 구성
X_attack = np.array(member_features + nonmember_features)
y_attack = np.array(member_labels + nonmember_labels)
print(f"공격 모델 학습용 데이터 차원: {X_attack.shape}, {y_attack.shape}")
# 출력 예: 공격 모델 학습용 데이터 차원: (12500, 2) (12500,)

# 간단한 MLP 공격 모델 정의
attack_model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
).to(device)
attack_optimizer = optim.Adam(attack_model.parameters(), lr=1e-3)
attack_criterion = nn.CrossEntropyLoss()

# 공격 모델 학습
attack_model.train()
X_tensor = torch.tensor(X_attack, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_attack, dtype=torch.long).to(device)
for epoch in range(50):  # 50 epoch 정도 학습
    # 순전파
    logits = attack_model(X_tensor)
    loss = attack_criterion(logits, y_tensor)
    # 역전파
    attack_optimizer.zero_grad()
    loss.backward()
    attack_optimizer.step()
# 학습 완료 - 최종 손실 출력
print(f"공격 모델 학습 완료 (최종 손실={loss.item():.4f})")

# =====================================
# 8. MIA 공격 평가 (w/o DP)
# =====================================
print("\n[비보호 버전] MIA 공격 평가 중...")

# 타겟 모델 출력 수집
target_model = model  # 이전 단계에서 학습한 타겟 모델 (비DP)
target_model.eval()

# 타겟 모델의 Part A (멤버) 출력 수집
member_scores = []
for batch in eval_loader_A2:
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        outputs = target_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    member_scores.extend(probs.cpu().tolist())

# 타겟 모델의 Part C (비멤버) 출력 수집
nonmember_scores = []
for batch in eval_loader_B3:
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        outputs = target_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    nonmember_scores.extend(probs.cpu().tolist())

# 공격 모델을 사용하여 멤버십 예측
attack_model.eval()
scores_all = np.array(member_scores + nonmember_scores, dtype=np.float32)
with torch.no_grad():
    attack_logits = attack_model(torch.tensor(scores_all).to(device))
    attack_probs = torch.softmax(attack_logits, dim=1)[:, 1].cpu().numpy()  # 예측을 멤버일 확률로

# 실제 라벨 배열 구성 (멤버=1 len(A)개 + 비멤버=0 len(C)개)
y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))

# ROC 커브 및 AUC 계산
fpr, tpr, thresholds = metrics.roc_curve(y_true, attack_probs)
auc_score_no_dp = metrics.roc_auc_score(y_true, attack_probs)
acc_attack_no_dp = metrics.accuracy_score(y_true, (attack_probs >= 0.5).astype(int))

print(f"공격 모델 AUC: {auc_score_no_dp:.4f}, 정확도(Thr=0.5): {acc_attack_no_dp:.4f}")
print(f"첫 5개 FPR: {fpr[:5]}, 첫 5개 TPR: {tpr[:5]}")

print(f"[비보호 타겟] Shadow+MLP MIA - 정확도: {acc_attack_no_dp:.4f}, AUC: {auc_score_no_dp:.4f}")

# SimpleConf MIA (간단한 신뢰도 기반 공격)
member_conf = [max(s) for s in member_scores]
nonmember_conf = [max(s) for s in nonmember_scores]
conf_scores = np.array(member_conf + nonmember_conf)
simple_auc_no_dp = metrics.roc_auc_score(y_true, conf_scores)
print(f"[비보호 타겟] SimpleConf MIA - AUC: {simple_auc_no_dp:.4f}")

# =====================================
# 9. DistilBERT 파인튜닝 (w/ DP) - DP 타겟 모델
# =====================================
print("\n" + "="*60)
print("[DP 보호 타겟 모델] DistilBERT 파인튜닝 (w/ DP)")
print("="*60)

# DP 적용을 위한 새로운 모델 초기화 (타겟 모델과 동일 구조, Part A에 대해 학습할 것)
model_dp = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# Opacus 호환성을 위해 모델 수정 (LayerNorm 등)
model_dp = ModuleValidator.fix(model_dp)
model_dp.to(device)
model_dp.train()  # 중요: Opacus는 training mode 필수!

optimizer_dp = optim.AdamW(model_dp.parameters(), lr=2e-5)

# PrivacyEngine 설정 및 부착
noise_multiplier = 1.0   # TODO: 노이즈 수준 조정 가능 (예: 0.5 ~ 1.0 사이)
max_grad_norm = 1.0      # TODO: 클리핑 임계값 설정 (작을수록 강한 클리핑)

# DP용 DataLoader (drop_last=True 필수)
train_loader_A1_dp = DataLoader(train_A1_tok, batch_size=batch_size, shuffle=True, 
                                 collate_fn=data_collator, drop_last=True)

privacy_engine = PrivacyEngine()
model_dp, optimizer_dp, train_A_loader_dp = privacy_engine.make_private(
    module=model_dp,
    optimizer=optimizer_dp,
    data_loader=train_loader_A1_dp,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm
)

# 학습 루프 (w/ DP)
epochs_dp = 1   # DP 모델 학습 epoch (과적합 방지를 위해 짧게 설정)
delta = 1e-5    # delta (데이터셋의 크기에 따라 1e-5 혹은 1e-6 자주 사용)

for epoch in range(epochs_dp):
    model_dp.train()
    total_loss = 0
    for batch in tqdm(train_A_loader_dp, desc=f"DP Training Epoch {epoch+1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        optimizer_dp.zero_grad()
        outputs = model_dp(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_dp.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_A_loader_dp)
    # Epoch 종료 후 누적 (ε) 계산
    epsilon = privacy_engine.get_epsilon(delta=delta)
    print(f"Epoch {epoch+1} DP-SGD 완료 - Privacy budget ε = {epsilon:.2f} (δ={delta})")

# DP 타겟 모델 평가 (분류 정확도)
model_dp.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_A2:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        outputs = model_dp(**inputs)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_dp = correct / total
print(f"[DP 타겟] 분류 정확도 - A2(holdout): {acc_target_dp:.4f}")

# B3 평가
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_B3:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        outputs = model_dp(**inputs)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_dp_b3 = correct / total
print(f"[DP 타겟] 분류 정확도 - B3(nonmember): {acc_target_dp_b3:.4f}")

# =====================================
# 10. 새도우 모델 기반 MIA 구현 및 평가 (w/ DP 용)
# =====================================
print("\n" + "="*60)
print("[DP 버전] 새도우 모델 기반 MIA 구현 및 평가")
print("="*60)

# DP 적용 새도우 모델
shadow_model_dp = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# Opacus 호환성을 위해 모델 수정
shadow_model_dp = ModuleValidator.fix(shadow_model_dp)
shadow_model_dp.to(device)
shadow_model_dp.train()  # 중요: Opacus는 training mode 필수!

shadow_optimizer_dp = optim.AdamW(shadow_model_dp.parameters(), lr=2e-5)

# DP용 DataLoader
train_loader_B1_dp = DataLoader(test_B1_tok, batch_size=batch_size, shuffle=True, 
                                 collate_fn=data_collator, drop_last=True)

privacy_engine_shadow = PrivacyEngine()
shadow_model_dp, shadow_optimizer_dp, train_B1_loader_dp = privacy_engine_shadow.make_private(
    module=shadow_model_dp,
    optimizer=shadow_optimizer_dp,
    data_loader=train_loader_B1_dp,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm
)

# 새도우 모델 DP-SGD 학습 (간략히 1 epoch)
print("\n[DP] 새도우 모델 학습 중...")
for batch in tqdm(train_B1_loader_dp, desc="Shadow Model (DP) Training"):
    shadow_optimizer_dp.zero_grad()
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    labels = batch["labels"].to(device)
    outputs = shadow_model_dp(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    shadow_optimizer_dp.step()

# DP 새도우 모델의 멤버/비멤버 출력 수집
shadow_model_dp.eval()
member_feat_dp = []
nonmember_feat_dp = []

with torch.no_grad():
    for batch in train_loader_B1:
        outputs = shadow_model_dp(**{k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        probs = torch.softmax(outputs.logits, dim=1)
        member_feat_dp.extend(probs.cpu().tolist())
    
    for batch in eval_loader_B2:
        outputs = shadow_model_dp(**{k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        probs = torch.softmax(outputs.logits, dim=1)
        nonmember_feat_dp.extend(probs.cpu().tolist())

X_attack_dp = np.array(member_feat_dp + nonmember_feat_dp)
y_attack_dp = np.array([1] * len(member_feat_dp) + [0] * len(nonmember_feat_dp))

# 공격 모델 학습 (동일 구조 MLP 사용 가능)
attack_model_dp = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2)
).to(device)
optim_attack_dp = optim.Adam(attack_model_dp.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
X_t = torch.tensor(X_attack_dp, dtype=torch.float32).to(device)
y_t = torch.tensor(y_attack_dp, dtype=torch.long).to(device)

attack_model_dp.train()
for epoch in range(50):
    logits = attack_model_dp(X_t)
    loss = loss_fn(logits, y_t)
    optim_attack_dp.zero_grad()
    loss.backward()
    optim_attack_dp.step()

print(f"[DP] 공격 모델 학습 완료 (최종 손실={loss.item():.4f})")

# =====================================
# 11. MIA 공격 평가 (w/ DP)
# =====================================
print("\n[DP 버전] MIA 공격 평가 중...")

# DP 타겟 모델의 출력 수집
model_dp.eval()
member_scores_dp = []
nonmember_scores_dp = []

with torch.no_grad():
    for batch in eval_loader_A2:
        outputs = model_dp(**{k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        probs = torch.softmax(outputs.logits, dim=1)
        member_scores_dp.extend(probs.cpu().tolist())
    
    for batch in eval_loader_B3:
        outputs = model_dp(**{k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        probs = torch.softmax(outputs.logits, dim=1)
        nonmember_scores_dp.extend(probs.cpu().tolist())

# MIA 공격 평가
attack_model_dp.eval()
scores_all_dp = np.array(member_scores_dp + nonmember_scores_dp, dtype=np.float32)
y_true_dp = np.array([1] * len(member_scores_dp) + [0] * len(nonmember_scores_dp))

with torch.no_grad():
    attack_logits_dp = attack_model_dp(torch.tensor(scores_all_dp).to(device))
    attack_probs_dp = torch.softmax(attack_logits_dp, dim=1)[:, 1].cpu().numpy()

auc_score_dp = metrics.roc_auc_score(y_true_dp, attack_probs_dp)
acc_attack_dp = metrics.accuracy_score(y_true_dp, (attack_probs_dp >= 0.5).astype(int))
print(f"[DP 타겟] Shadow+MLP MIA - 정확도: {acc_attack_dp:.4f}, AUC: {auc_score_dp:.4f}")

# SimpleConf MIA (DP 버전)
member_conf_dp = [max(s) for s in member_scores_dp]
nonmember_conf_dp = [max(s) for s in nonmember_scores_dp]
conf_scores_dp = np.array(member_conf_dp + nonmember_conf_dp)
simple_auc_dp = metrics.roc_auc_score(y_true_dp, conf_scores_dp)
print(f"[DP 타겟] SimpleConf MIA - AUC: {simple_auc_dp:.4f}")

# =====================================
# 12. 최종 결과 비교
# =====================================
print("\n" + "="*60)
print("=== 최종 요약 ===")
print("="*60)

print(f"\n[비보호 타겟] 분류 - A2(holdout): {acc_target_no_dp:.4f}, B3(nonmember): {acc_target_no_dp_b3:.4f}")
print(f"[DP 타겟]     분류 - A2(holdout): {acc_target_dp:.4f}, B3(nonmember): {acc_target_dp_b3:.4f}")
print(f"\n[비보호 타겟] Shadow+MLP MIA  - 정확도: {acc_attack_no_dp:.4f}, AUC: {auc_score_no_dp:.4f}")
print(f"[DP 타겟]     Shadow+MLP MIA  - 정확도: {acc_attack_dp:.4f}, AUC: {auc_score_dp:.4f}")
print(f"\n[비보호 타겟] SimpleConf MIA  - AUC: {simple_auc_no_dp:.4f}")
print(f"[DP 타겟]     SimpleConf MIA  - AUC: {simple_auc_dp:.4f}")

# Privacy budget
final_epsilon = privacy_engine.get_epsilon(delta=delta)
print(f"\n[DP 타겟]     epsilon(): {final_epsilon:.2f}, delta={delta} (설정값)")

print("\n" + "="*60)
print("분석 결론:")
print("="*60)

if simple_auc_dp < simple_auc_no_dp:
    print(f"✓ DP-SGD 적용으로 SimpleConf MIA AUC가 {simple_auc_no_dp:.4f} → {simple_auc_dp:.4f}로 감소")
    print(f"  프라이버시가 향상됨 (0.5에 가까울수록 랜덤 추측 수준)")
else:
    print("✗ DP-SGD 적용 후에도 MIA AUC가 감소하지 않음")

if acc_target_dp < acc_target_no_dp:
    acc_drop = ((acc_target_no_dp - acc_target_dp) / acc_target_no_dp) * 100 if acc_target_no_dp > 0 else 0
    print(f"! 단, 모델 정확도가 {acc_target_no_dp:.4f} → {acc_target_dp:.4f}로 {acc_drop:.1f}% 하락")
    print(f"  (privacy-utility trade-off)")
else:
    print("! 모델 정확도는 유지 또는 향상됨")

print("\n실험 완료!")
