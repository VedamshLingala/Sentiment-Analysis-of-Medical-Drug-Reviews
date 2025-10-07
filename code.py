# ===============================================
# 1️⃣ Imports
# ===============================================
import pandas as pd
import re
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import numpy as np
import os

# LoRA imports
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===============================================
# 2️⃣ Mount Google Drive
# ===============================================
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/roberta_LoRA_DrugReview_Model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================================
# 3️⃣ Load dataset (skip bad lines)
# ===============================================
train_df = pd.read_csv('drugsComTrain_raw.csv', on_bad_lines='skip')
test_df  = pd.read_csv('drugsComTest_raw.csv', on_bad_lines='skip')

# ===============================================
# 4️⃣ Basic text cleaning
# ===============================================
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

train_df['cleaned_text'] = train_df['review'].apply(clean_text)
test_df['cleaned_text']  = test_df['review'].apply(clean_text)

# ===============================================
# 5️⃣ Map ratings into 3 sentiment classes
# ===============================================
def group_rating(r):
    if r <= 3:
        return 0
    elif r <= 7:
        return 1
    else:
        return 2

train_df['label'] = train_df['rating'].apply(group_rating)
test_df['label']  = test_df['rating'].apply(group_rating)

# ===============================================
# 6️⃣ Train-validation split
# ===============================================
xtrain, xval, ytrain, yval = train_test_split(
    train_df['cleaned_text'], train_df['label'], test_size=0.1, random_state=42
)

# ===============================================
# 7️⃣ Tokenizer
# ===============================================
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

train_encodings = tokenizer(list(xtrain), truncation=True, padding=True, max_length=100)
val_encodings   = tokenizer(list(xval), truncation=True, padding=True, max_length=100)
test_encodings  = tokenizer(list(test_df['cleaned_text']), truncation=True, padding=True, max_length=100)

# ===============================================
# 8️⃣ Dataset wrapper
# ===============================================
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.tolist()) if labels is not None else None
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

train_dataset = ReviewDataset(train_encodings, ytrain)
val_dataset   = ReviewDataset(val_encodings, yval)
test_dataset  = ReviewDataset(test_encodings, test_df['label'])

# ===============================================
# 9️⃣ DataLoaders
# ===============================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16, collate_fn=data_collator)
val_loader   = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16, collate_fn=data_collator)
test_loader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16, collate_fn=data_collator)

# ===============================================
# 🔟 Load RoBERTa Model and Apply LoRA
# ===============================================
base_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)

# LoRA Configuration (optimal setup)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "value"],  # focus on attention projections
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===============================================
# 1️⃣1️⃣ Fine-tuning with LoRA
# ===============================================
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} (LoRA Fine-tuning)", colour='blue')

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)
        loop.set_postfix(loss=loss.item(), acc=correct/total)

    print(f"\nEpoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total:.4f}")

    # Save LoRA fine-tuned model
    model.save_pretrained(f"{SAVE_DIR}/roberta_lora_epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{SAVE_DIR}/roberta_lora_epoch_{epoch+1}")
    print(f"💾 LoRA Model saved to {SAVE_DIR}/roberta_lora_epoch_{epoch+1}")

# ===============================================
# 1️⃣2️⃣ Evaluate LoRA Fine-tuned Model
# ===============================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    loop = tqdm(test_loader, desc="Evaluating LoRA-RoBERTa", colour='green')
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

lora_acc = accuracy_score(all_labels, all_preds)
lora_cm = confusion_matrix(all_labels, all_preds)

print(f"\n🟢 LoRA Fine-tuned RoBERTa Accuracy: {lora_acc*100:.2f}%")
print("🟢 Confusion Matrix:\n", lora_cm)
