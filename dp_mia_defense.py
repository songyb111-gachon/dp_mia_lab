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
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from datasets import load_dataset, concatenate_datasets
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
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
# 커스텀 Dataset 클래스 (고정 길이)
# =====================================
class IMDbDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

# =====================================
# 2. IMDb 데이터셋 불러오기
# =====================================
print("\n[Step 1] IMDb 데이터셋 로드 중...")
imdb = load_dataset("imdb")
train_full = imdb["train"]
test_full = imdb["test"]

print(f"전체 훈련 샘플 수: {len(train_full)}, 전체 테스트 샘플 수: {len(test_full)}")

# =====================================
# 3. 데이터셋 분할 (셔플 후 분할!)
# =====================================
print("\n[Step 2] 데이터셋 분할 중...")

# 중요: IMDb는 정렬되어 있으므로 셔플 필요!
train_full_shuffled = train_full.shuffle(seed=42)
test_full_shuffled = test_full.shuffle(seed=42)

# A1 (멤버 학습), A2 (멤버 테스트용)
train_A1 = train_full_shuffled.select(range(12500))
train_A2 = train_full_shuffled.select(range(12500, 25000))

# B1, B2, B3: 테스트셋 분할
test_B1 = test_full_shuffled.select(range(6250))
test_B2 = test_full_shuffled.select(range(6250, 12500))
test_B3 = test_full_shuffled.select(range(12500, 25000))

print(f"A1 (타겟 학습): {len(train_A1)}, A2 (MIA 멤버): {len(train_A2)}")
print(f"B1 (새도우 학습): {len(test_B1)}, B2 (새도우 비멤버): {len(test_B2)}, B3 (MIA 비멤버): {len(test_B3)}")

# 라벨 분포 확인
print(f"[디버그] A1 라벨 분포: 0={sum(1 for x in train_A1['label'] if x==0)}, 1={sum(1 for x in train_A1['label'] if x==1)}")
print(f"[디버그] A2 라벨 분포: 0={sum(1 for x in train_A2['label'] if x==0)}, 1={sum(1 for x in train_A2['label'] if x==1)}")

# =====================================
# 4. 토큰화 (고정 길이 - Opacus 호환)
# =====================================
print("\n[Step 3] 토큰화 중...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def create_dataset(hf_dataset):
    """HuggingFace 데이터셋을 PyTorch Dataset으로 변환 (고정 길이)"""
    texts = list(hf_dataset['text'])
    labels = list(hf_dataset['label'])
    
    # 고정 길이 패딩 (Opacus 호환)
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding='max_length',  # 고정 길이!
        max_length=128, 
        return_tensors='pt'
    )
    
    return IMDbDataset(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
        labels=torch.tensor(labels, dtype=torch.long)
    )

print("  토큰화 진행 중...")
dataset_A1 = create_dataset(train_A1)
dataset_A2 = create_dataset(train_A2)
dataset_B1 = create_dataset(test_B1)
dataset_B2 = create_dataset(test_B2)
dataset_B3 = create_dataset(test_B3)

print(f"  A1: {len(dataset_A1)}, A2: {len(dataset_A2)}, B1: {len(dataset_B1)}, B2: {len(dataset_B2)}, B3: {len(dataset_B3)}")

# =====================================
# 5. Dataloader 준비
# =====================================
print("\n[Step 4] Dataloader 준비 중...")
batch_size = 16

train_loader_A1 = DataLoader(dataset_A1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
eval_loader_A2 = DataLoader(dataset_A2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
eval_loader_B3 = DataLoader(dataset_B3, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_loader_B1 = DataLoader(dataset_B1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
eval_loader_B2 = DataLoader(dataset_B2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"배치 사이즈: {batch_size}")

# =====================================
# 6. DistilBERT 파인튜닝 (w/o DP) - 타겟 모델
# =====================================
print("\n" + "="*60)
print("[비보호 타겟 모델] DistilBERT 파인튜닝 (w/o DP)")
print("="*60)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

epochs = 1
model.train()
for epoch in range(epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch in tqdm(train_loader_A1, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = outputs.logits.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    
    avg_loss = total_loss / len(train_loader_A1)
    train_acc = correct_train / total_train
    print(f"Epoch {epoch+1} 완료 - 평균 손실: {avg_loss:.4f}, 학습 정확도: {train_acc:.4f}")

# 타겟 모델 평가
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_A2:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_no_dp = correct / total
print(f"[비보호 타겟] 분류 정확도 - A2(holdout): {acc_target_no_dp:.4f}")

# B3 평가
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_B3:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

shadow_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)
shadow_optimizer = optim.AdamW(shadow_model.parameters(), lr=2e-5)

print("\n새도우 모델 학습 중...")
shadow_model.train()
for epoch in range(1):
    for batch in tqdm(train_loader_B1, desc="Shadow Model Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        shadow_optimizer.zero_grad()
        outputs = shadow_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        shadow_optimizer.step()

# Feature 수집
print("\nFeature 수집 중...")
shadow_model.eval()

member_features = []
for batch in train_loader_B1:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = shadow_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
    member_features.extend(probs.cpu().tolist())
member_labels = [1] * len(member_features)

nonmember_features = []
for batch in eval_loader_B2:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = shadow_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
    nonmember_features.extend(probs.cpu().tolist())
nonmember_labels = [0] * len(nonmember_features)

print(f"멤버 특징 개수: {len(member_features)}, 비멤버 특징 개수: {len(nonmember_features)}")

# 공격 모델 학습
print("\n공격 모델 학습 중...")
X_attack = np.array(member_features + nonmember_features)
y_attack = np.array(member_labels + nonmember_labels)
print(f"공격 모델 학습용 데이터 차원: {X_attack.shape}, {y_attack.shape}")

attack_model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
).to(device)
attack_optimizer = optim.Adam(attack_model.parameters(), lr=1e-3)
attack_criterion = nn.CrossEntropyLoss()

attack_model.train()
X_tensor = torch.tensor(X_attack, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_attack, dtype=torch.long).to(device)

for epoch in range(50):
    logits = attack_model(X_tensor)
    loss = attack_criterion(logits, y_tensor)
    attack_optimizer.zero_grad()
    loss.backward()
    attack_optimizer.step()

print(f"공격 모델 학습 완료 (최종 손실={loss.item():.4f})")

# =====================================
# 8. MIA 공격 평가 (w/o DP)
# =====================================
print("\n[비보호 버전] MIA 공격 평가 중...")

target_model = model
target_model.eval()

member_scores = []
for batch in eval_loader_A2:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = target_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
    member_scores.extend(probs.cpu().tolist())

nonmember_scores = []
for batch in eval_loader_B3:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = target_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
    nonmember_scores.extend(probs.cpu().tolist())

attack_model.eval()
scores_all = np.array(member_scores + nonmember_scores, dtype=np.float32)
with torch.no_grad():
    attack_logits = attack_model(torch.tensor(scores_all).to(device))
    attack_probs = torch.softmax(attack_logits, dim=1)[:, 1].cpu().numpy()

y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))

auc_score_no_dp = metrics.roc_auc_score(y_true, attack_probs)
acc_attack_no_dp = metrics.accuracy_score(y_true, (attack_probs >= 0.5).astype(int))

print(f"[비보호 타겟] Shadow+MLP MIA - 정확도: {acc_attack_no_dp:.4f}, AUC: {auc_score_no_dp:.4f}")

# SimpleConf MIA
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

model_dp = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# Opacus 호환성을 위해 모델 수정
model_dp = ModuleValidator.fix(model_dp)
model_dp.to(device)
model_dp.train()  # 중요!

optimizer_dp = optim.AdamW(model_dp.parameters(), lr=2e-5)

# DP 설정
noise_multiplier = 1.0
max_grad_norm = 1.0

# DP용 DataLoader
train_loader_A1_dp = DataLoader(dataset_A1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

privacy_engine = PrivacyEngine()
model_dp, optimizer_dp, train_A_loader_dp = privacy_engine.make_private(
    module=model_dp,
    optimizer=optimizer_dp,
    data_loader=train_loader_A1_dp,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm
)

# 학습 루프 (w/ DP)
epochs_dp = 1
delta = 1e-5

for epoch in range(epochs_dp):
    model_dp.train()
    total_loss = 0
    for batch in tqdm(train_A_loader_dp, desc=f"DP Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer_dp.zero_grad()
        outputs = model_dp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_dp.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_A_loader_dp)
    epsilon = privacy_engine.get_epsilon(delta=delta)
    print(f"Epoch {epoch+1} DP-SGD 완료 - 평균 손실: {avg_loss:.4f}, ε = {epsilon:.2f} (δ={delta})")

# DP 타겟 모델 평가
model_dp.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_A2:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model_dp(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_target_dp = correct / total
print(f"[DP 타겟] 분류 정확도 - A2(holdout): {acc_target_dp:.4f}")

# B3 평가
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader_B3:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model_dp(input_ids=input_ids, attention_mask=attention_mask)
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

shadow_model_dp = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

shadow_model_dp = ModuleValidator.fix(shadow_model_dp)
shadow_model_dp.to(device)
shadow_model_dp.train()

shadow_optimizer_dp = optim.AdamW(shadow_model_dp.parameters(), lr=2e-5)

train_loader_B1_dp = DataLoader(dataset_B1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

privacy_engine_shadow = PrivacyEngine()
shadow_model_dp, shadow_optimizer_dp, train_B1_loader_dp = privacy_engine_shadow.make_private(
    module=shadow_model_dp,
    optimizer=shadow_optimizer_dp,
    data_loader=train_loader_B1_dp,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm
)

print("\n[DP] 새도우 모델 학습 중...")
shadow_model_dp.train()
for batch in tqdm(train_B1_loader_dp, desc="Shadow Model (DP) Training"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    shadow_optimizer_dp.zero_grad()
    outputs = shadow_model_dp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    shadow_optimizer_dp.step()

# DP 새도우 모델의 멤버/비멤버 출력 수집
shadow_model_dp.eval()
member_feat_dp = []
nonmember_feat_dp = []

with torch.no_grad():
    for batch in train_loader_B1:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = shadow_model_dp(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        member_feat_dp.extend(probs.cpu().tolist())
    
    for batch in eval_loader_B2:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = shadow_model_dp(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        nonmember_feat_dp.extend(probs.cpu().tolist())

X_attack_dp = np.array(member_feat_dp + nonmember_feat_dp)
y_attack_dp = np.array([1] * len(member_feat_dp) + [0] * len(nonmember_feat_dp))

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

model_dp.eval()
member_scores_dp = []
nonmember_scores_dp = []

with torch.no_grad():
    for batch in eval_loader_A2:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model_dp(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        member_scores_dp.extend(probs.cpu().tolist())
    
    for batch in eval_loader_B3:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model_dp(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        nonmember_scores_dp.extend(probs.cpu().tolist())

attack_model_dp.eval()
scores_all_dp = np.array(member_scores_dp + nonmember_scores_dp, dtype=np.float32)
y_true_dp = np.array([1] * len(member_scores_dp) + [0] * len(nonmember_scores_dp))

with torch.no_grad():
    attack_logits_dp = attack_model_dp(torch.tensor(scores_all_dp).to(device))
    attack_probs_dp = torch.softmax(attack_logits_dp, dim=1)[:, 1].cpu().numpy()

auc_score_dp = metrics.roc_auc_score(y_true_dp, attack_probs_dp)
acc_attack_dp = metrics.accuracy_score(y_true_dp, (attack_probs_dp >= 0.5).astype(int))
print(f"[DP 타겟] Shadow+MLP MIA - 정확도: {acc_attack_dp:.4f}, AUC: {auc_score_dp:.4f}")

# SimpleConf MIA (DP)
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
