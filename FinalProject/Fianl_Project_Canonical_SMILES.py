from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 모델 및 토크나이저 로드
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)

# 데이터 로드
train_data = pd.read_csv("./Dataset/qm9_canonical_train.csv")
test_data = pd.read_csv("./Dataset/qm9_canonical_test.csv")

# SMILES 토큰화
class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, target_col):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['canonical_smiles']
        target = self.data.iloc[idx][self.target_col]
        tokens = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in tokens.items()}
        item['labels'] = torch.tensor(target, dtype=torch.float)
        return item

# 데이터셋 준비 (전체 데이터 사용)
train_dataset = SMILESDataset(train_data, tokenizer, 'gap')
test_dataset = SMILESDataset(test_data, tokenizer, 'gap')

# 데이터 Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 학습 파라미터
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # 배치 크기
    per_device_eval_batch_size=16,
    num_train_epochs=2,  # 에포크 수
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained("./jjh/ChemBERTa-canonical_smiles-gap")
tokenizer.save_pretrained("./jjh/ChemBERTa-canonical_smiles-gap")

print("파인튜닝 완료 및 모델 저장 완료!")

# 테스트 데이터 평가
evaluation_results = trainer.evaluate()
print("테스트 데이터 평가 결과:", evaluation_results)
