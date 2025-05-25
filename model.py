# Install necessary libraries (only needed if not already available)
!pip install torch torchvision pandas scikit-learn pillow tqdm

import os
import pandas as pd

# Correct paths for Kaggle
TRAIN_DIR = "../input/soil-dataset/train/train"
TEST_DIR = "../input/soil-dataset/test/test"
TRAIN_CSV = "../input/soil-dataset/train_labels.csv"

# Load CSV
df = pd.read_csv(TRAIN_CSV)

# Print some info to verify
print("Train images:", len(os.listdir(TRAIN_DIR)))
print("Test images:", len(os.listdir(TEST_DIR)))
print("CSV preview:")
print(df.head())

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score

# Set up GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CSV
df = pd.read_csv(TRAIN_CSV)
class_names = df['soil_type'].unique().tolist()
class_to_idx = {label: idx for idx, label in enumerate(class_names)}
df['label'] = df['soil_type'].map(class_to_idx)

# Image transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset
class SoilDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'])
        image = Image.open(image_path).convert("RGB")
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Train/Val split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_dataset = SoilDataset(train_df, TRAIN_DIR, transform=train_transform)
val_dataset = SoilDataset(val_df, TRAIN_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, val_loader)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))))
    for idx, score in enumerate(f1_per_class):
        print(f"{class_names[idx]} F1-score: {score:.4f}")
    print(f"Minimum F1-score: {min(f1_per_class):.4f}\n")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# Save the model
torch.save(model.state_dict(), "soil_classifier_resnet18.pth")

class TestDataset(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_id

# Load test image names
test_image_list = sorted(os.listdir(TEST_DIR))
test_dataset = TestDataset(TEST_DIR, test_image_list, val_transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Run inference
model.eval()
predictions = []

with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        for img_id, label in zip(image_ids, preds):
            predictions.append((img_id, class_names[label]))

# Save to CSV
submission_df = pd.DataFrame(predictions, columns=["image_id", "soil_type"])
submission_df.to_csv("submission.csv", index=False)
print("Saved predictions to submission.csv")
