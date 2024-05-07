
import numpy as np
from dataset import Dataset_ske
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import HeavyPoseClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

# sample command : 
# python3 torch_heavy_pose_classifier.py --run mixedfullswap --n_epochs 100  --train /ske/data/kp_16_cover_modes/mixed/fullswaptrainmixed.csv --test /ske/data/kp_16_cover_modes/mixed/testmixed.csv --val /ske/data/kp_16_cover_modes/mixed/testmixed.csv
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, help="the name of the run")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--drop_p", type=float, default=0.3, help="drop_out rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--train",type=str, help="path to train csv")
parser.add_argument("--test",type=str, help="path to test csv")
parser.add_argument("--val",type=str, help="path to validation csv")
opt = parser.parse_args()
print(opt)

output_dir="output/heavy_pose_classifier"
output_run = os.path.join(output_dir,opt.run)
checkpoint_save_dir=os.path.join(output_run,"checkpoints")
graph_save_dir=os.path.join(output_run,"graphs")
if not os.path.exists(graph_save_dir): os.makedirs(graph_save_dir)
if not os.path.exists(checkpoint_save_dir): os.makedirs(checkpoint_save_dir)

trainset=Dataset_ske(opt.train)
valset=Dataset_ske(opt.test)
testset=Dataset_ske(opt.test)


trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0
)
valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0
)
testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0
)
net = HeavyPoseClassifier(drop_out_p=opt.drop_p)
#net = SimplePoseClassifier()
num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
num_params = sum(p.numel() for p in net.parameters())
print("Trainable param : {} total param : {}".format(num_trainable_params,num_params))
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
net.to(device)
print("==> Total C parameters: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
criterion = nn.CrossEntropyLoss()
#criterion = AdMSoftmaxLoss(9, 9, s=30.0, m=0.4) 
optimizer = optim.Adam(net.parameters(), opt.lr)
# model_state, optimizer_state = torch.load(os.path.join("runs\kinetic-gan\classi_models_finetune", "checkpoint_ep86_iterloss0.0_valloss0.7235625386238098_valacc0.8424289008455035.pt"))
# net.load_state_dict(model_state)
# optimizer.load_state_dict(optimizer_state)
train_loss_values=[]
val_acc_values=[]
val_loss_values=[]
best_val_acc=0.0
best_ckpt_metric={"epoch":0,"iter_loss":0.0,"val_loss":0.0,"val_acc":0.0}
for epoch in range(opt.n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs)
        #print(inputs)
        inputs, labels = inputs.float().to(device), labels.type(torch.LongTensor).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        # print(labels)
        #print(outputs.get_device())
        #print(labels.get_device())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1
    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
            

    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.type(torch.LongTensor).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1
            
    val_loss_values.append(float(val_loss / val_steps))
    val_acc_values.append(float(correct/total))
    train_loss_values.append(float(running_loss / epoch_steps))
    print("[%d, %5d] val_loss: %.3f accuracy: %.3f"  % (epoch + 1, i + 1,val_loss / val_steps,correct/total))

    # Here we save a checkpoint. It is automatically registered with
    # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
    # in future iterations.
    # Note to save a file like checkpoint, you still need to put it under a directory
    # to construct a checkpoint.
    if (correct/total>best_val_acc):
        best_val_acc=correct/total
                
        # torch.save(
        #     (net.state_dict(), optimizer.state_dict()), os.path.join("runs/kinetic-gan/classi_models", "checkpoint_ep{}_iterloss{}_valloss{}_valacc{}.pt".format(epoch,str(running_loss/epoch_steps),str(val_loss/val_steps),str(correct/total)))
        # )
        best_ckpt_metric["epoch"]=epoch
        best_ckpt_metric["iter_loss"]=running_loss/epoch_steps
        best_ckpt_metric["val_loss"]=val_loss/val_steps
        best_ckpt_metric["val_acc"]=correct/total
        torch.save((net.state_dict()),os.path.join(checkpoint_save_dir,"best_ckpt.pt"))
    running_loss = 0.0
os.rename(os.path.join(checkpoint_save_dir,"best_ckpt.pt"), os.path.join(checkpoint_save_dir, "best_ckpt_ep{}_iterloss{}_valloss{}_valacc{}.pt".format(best_ckpt_metric["epoch"],str(best_ckpt_metric["iter_loss"]),str(best_ckpt_metric["val_loss"]),str(best_ckpt_metric["val_acc"]))))
print("Finished epoch {}".format(epoch))
print("Best ckpt metrics {}".format(best_ckpt_metric))



# Calculate accuracy for each class
class_correct = [0] * 9
class_total = [0] * 9
# Calculate misclassification between classes
classification_table = np.zeros((9, 9), dtype=int)
for i, data in enumerate(valloader, 0):
    with torch.no_grad():
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.type(torch.LongTensor).to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # Calculate accuracy for each class
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

        for j in range(len(labels)):
            true_label = labels[j]
            predicted_label = predicted[j]
            classification_table[true_label][predicted_label] += 1

# Calculate accuracy for each class
for i in range(9):
    accuracy = 100 * class_correct[i] / class_total[i]
    print(f"Class {i}: Accuracy = {accuracy:.2f}%")
# calculating and display the classification table
percent_classification_table=classification_table
for i in range(9):
    for j in range(9):
        percent_classification_table[i][j]=percent_classification_table[i][j]*100/class_total[i]
headers = [f"Predicted {i}" for i in range(9)]
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.table(cellText=percent_classification_table, colLabels=headers, loc='center', cellLoc='center')
plt.savefig(os.path.join(graph_save_dir,"confusion_matrix.png"))
plt.close()

    



import matplotlib.pyplot as plt

# ... (Training loop and loss calculation)

# Plot loss curve
# print("train_loss_values: ")
# print(train_loss_values)
plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'train_loss_curve.png'))
plt.close()

# print("val_acc_values: ")
# print(val_acc_values)
plt.plot(val_acc_values, label='Val acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('Val Acc Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'val_acc_curve.png'))
plt.close()

# print("val_loss_values: ")
# print(val_loss_values)
plt.plot(val_loss_values, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Val Loss Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'val_loss_curve.png'))
plt.close()