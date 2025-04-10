import os
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# 1. Extraction: Update these paths to match your local PC file locations.
# Path to your local zip file
zip_path = 'kaggle/input/archive.zip'  # <-- Update this path
# Directory where the contents will be extracted
extraction_dir = 'kaggle/input/'  # <-- Update this path

# Create extraction directory if it doesn't exist
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# 2. Dataset Directories: Adjust these paths based on your extracted dataset.
train_dir = os.path.join(extraction_dir, 'filtered_DATASET', 'TRAIN')  # Change if needed
test_dir  = os.path.join(extraction_dir, 'filtered_DATASET', 'TEST')   # Change if needed

# 3. Data Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 4. Custom Dataset Class
class WasteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Using sorted() ensures that label ordering is consistent
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 5. Create Datasets and DataLoaders
train_dataset = WasteDataset(train_dir, transform=transform_train)
test_dataset = WasteDataset(test_dir, transform=transform_test)

# When using a GPU like the RTX 4050, pin_memory=True can speed up host-to-device transfers.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# 6. Model Definition: EfficientNetV2Lite
class EfficientNetV2Lite(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2Lite, self).__init__()
        # For PyTorch >= 1.13, you can use the new weights API:
        self.features = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        ).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 7. Setup Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)  # Dynamic class count based on dataset
model = EfficientNetV2Lite(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_acc = 0.0

# 8. Training Loop with tqdm and Checkpointing
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training with tqdm progress bar
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    # Save checkpoint if accuracy improves
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"--> New best model saved with accuracy: {accuracy:.2f}%")

# Save final model after training
torch.save(model.state_dict(), 'final_model.pth')
print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
