from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torchvision
import torch
import shutil
from tqdm import tqdm
from dataset import IRDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse
import os
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_model", type=str, default="model", help="output model, output_root/output_model")
parser.add_argument("--test_dir", type=str, help="inside datadir will be 1 -> 9 subfolder")
opt = parser.parse_args()
print(opt)
# dataset
testset = IRDataset(root_dir=opt.test_dir, transform=transform)
print(len(testset))
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
# loading ckpt
model = torchvision.models.efficientnet_b0(pretrained=False)
state=torch.load(opt.ckpt_model)
model.load_state_dict(state["state_dict"])
model.to(device)
# Loss Function
criterion = nn.CrossEntropyLoss()
# Validation function
def validate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
print(validate(model,test_loader,criterion))