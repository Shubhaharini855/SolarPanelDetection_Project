# src/train_resnet.py
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\HP\rooftop_pv_pipeline\dataset_resnet"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "models"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "resnet50_best.pth")
LOG_JSON = os.path.join(SAVE_DIR, "resnet_training_log.json")
os.makedirs(SAVE_DIR, exist_ok=True)
# ----------------------------------------

# -------------- FAST TRANSFORMS ----------------
# (NO RandomResizedCrop → very slow on CPU)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Faster & more stable
    transforms.RandomHorizontalFlip(),  # Light augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# ------------------------------------------------

# Datasets & loaders
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

print(f"Classes: {train_ds.classes}")
print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# ------------------ MODEL ----------------------
# Fix deprecated "pretrained=True"
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

# ------------------- TRAINING LOOP -------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training")

    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())

    epoch_train_loss = running_loss / len(train_ds)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss = val_loss / len(val_ds)
    epoch_val_acc = correct / total
    scheduler.step(epoch_val_acc)

    print(f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.4f}")

    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    # -------- SAVE BEST MODEL --------
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved best model -> {BEST_MODEL_PATH} (val_acc={best_val_acc:.4f})")

# Save training log
with open(LOG_JSON, "w") as f:
    json.dump(history, f, indent=2)

print("Training complete. Best val acc:", best_val_acc)
