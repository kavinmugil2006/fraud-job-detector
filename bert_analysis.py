# bert_analysis.py (GPU Optimized)

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# =============================
# Settings (GPU POWER 🔥)
# =============================

DATA_PATH = r"C:\Users\namik\OneDrive\Desktop\fakejobdetector\fake_job_postings.csv"

SAMPLE_SIZE = 30000       # Increase if GPU is strong
BATCH_SIZE = 32           # GPU friendly
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256


# =============================
# Device
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if device.type != "cuda":
    print("⚠️ WARNING: GPU NOT DETECTED!")


# =============================
# Load Dataset
# =============================

data = pd.read_csv(DATA_PATH)

# Auto detect columns
TEXT_COL = None
LABEL_COL = None

for col in data.columns:
    name = col.lower()

    if "text" in name or "desc" in name or "job" in name:
        TEXT_COL = col

    if "label" in name or "fraud" in name or "fake" in name:
        LABEL_COL = col


data = data[[TEXT_COL, LABEL_COL]]
data.dropna(inplace=True)
data.columns = ["text", "label"]


# Sample (for speed)
data = data.sample(SAMPLE_SIZE, random_state=42)

print("Using rows:", len(data))


# =============================
# Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# =============================
# Tokenizer
# =============================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =============================
# Dataset
# =============================

class JobDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =============================
# Loaders (GPU tuned)
# =============================

train_ds = JobDataset(X_train, y_train)
test_ds = JobDataset(X_test, y_test)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True
)


# =============================
# Model
# =============================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# =============================
# Training
# =============================

print("\n🔥 Training BERT on GPU...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Avg Loss:", total_loss / len(train_loader))


# =============================
# Evaluation
# =============================

print("\nEvaluating...")

model.eval()

preds = []
true = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask
        )

        p = torch.argmax(outputs.logits, dim=1)

        preds.extend(p.cpu().numpy())
        true.extend(labels.cpu().numpy())


acc = accuracy_score(true, preds)

print("\n🔥 BERT GPU Accuracy:", round(acc * 100, 2), "% ❤️💙")
