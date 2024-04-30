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
# transform_train = transforms.Compose([
#     transforms.Resize((120, 160)), 
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

# transform_val = transforms.Compose([
#     transforms.Resize((120, 160)),
#     transforms.ToTensor(),
# ])

# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

parser = argparse.ArgumentParser()
parser.add_argument("--output_root", type=str, default="./output", help="output root")
parser.add_argument("--output_ckpt", type=str, default="checkpoint", help="output checkpoint, output_root/output_ckpt")
parser.add_argument("--output_model", type=str, default="model", help="output model, output_root/output_model")
parser.add_argument("--data_dir", type=str, help="inside datadir will be 1 -> 9 subfolder")
parser.add_argument("--train_dir", type=str, help="inside datadir will be 1 -> 9 subfolder")
parser.add_argument("--val_dir", type=str, help="inside datadir will be 1 -> 9 subfolder")
parser.add_argument("--test_dir", type=str, help="inside datadir will be 1 -> 9 subfolder")
parser.add_argument("--lr", type=float, default=0.002,help="learning rate")
parser.add_argument("--epochs",type=int,default=100,help="number of epoch")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output_root) : os.makedirs(opt.output_root)
if not os.path.exists(os.path.join(opt.output_root,opt.output_ckpt)) : os.makedirs(os.path.join(opt.output_root,opt.output_ckpt))
if not os.path.exists(os.path.join(opt.output_root,opt.output_model)) : os.makedirs(os.path.join(opt.output_root,opt.output_model))


""" From a big dataset, split !
dataset = IRDataset(root_dir=opt.data_dir, transform=transform)

train_indices, test_indices = train_test_split(
    list(range(dataset.__len__())),
    test_size=0.2, 
    stratify=dataset.get_labels()
)
print(len(train_indices))
print(len(test_indices))
# Dataloader
train_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0,sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0,sampler=SubsetRandomSampler(test_indices))
print(len(train_loader))
print(len(test_loader))
"""

""" We already have separate image folder for train and test"""
# dataset
trainset = IRDataset(root_dir=opt.train_dir, transform=transform)
testset = IRDataset(root_dir=opt.test_dir, transform=transform)
print(len(trainset))
print(len(testset))
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)


# Model
model = torchvision.models.efficientnet_b0(pretrained=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    savepath = os.path.join(opt.output_root,opt.output_ckpt,filename)
    torch.save(state, savepath)
    if is_best:
        bestpath=os.path.join(opt.output_root,opt.output_ckpt,"model_best.pt")
        shutil.copyfile(savepath, bestpath)

def save_model(model, filename='final_model.pt'):
    model_path=os.path.join(opt.output_root,opt.output_model,filename)
    torch.save(model.state_dict(), model_path)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar
            pbar.update(inputs.shape[0])
            pbar.set_postfix({'loss': running_loss / total, 'accuracy': 100. * correct / total})
        
        pbar.close()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, is_best)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
    print(f"The best acc is: {best_acc}")
    save_model(model, f'final_model_acc_{best_acc}.pt')

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
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

train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=opt.epochs)
