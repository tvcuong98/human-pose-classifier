
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
from AMSoftmax.AMSloss import AdMSoftmaxLoss as AMloss
output_dir="output/heavy_pose_classifier_AMloss"
checkpoint_save_dir=os.path.join(output_dir,"checkpoints")
graph_save_dir=os.path.join(output_dir,"graphs")
if not os.path.exists(graph_save_dir): os.makedirs(graph_save_dir)
if not os.path.exists(checkpoint_save_dir): os.makedirs(checkpoint_save_dir)

# Here is for normal training swap
trainset=Dataset_ske("../path_to_trainset")
valset=Dataset_ske("../path_to_valset")
testset=Dataset_ske("../path_to_testset")



trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=512,
    shuffle=True,
    num_workers=0
)
valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=512,
    shuffle=False,
    num_workers=0
)
testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=512,
    shuffle=False,
    num_workers=0
)
net = HeavyPoseClassifier(drop_out_p=0.3)
#net = SimplePoseClassifier()
num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
num_params = sum(p.numel() for p in net.parameters())
print("Trainable param : {} total param : {}".format(num_trainable_params,num_params))
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)

criterion = AMloss(9, 9, scale = 30.0, margin=0.4)
criterion.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)

train_loss_values=[]
val_acc_values=[]
val_loss_values=[]
best_val_acc=0.0
for epoch in range(50):  # loop over the dataset multiple times
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
        #print(outputs.get_device())
        #print(labels.get_device())
        loss = criterion(outputs, labels)
        #print(loss)
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
    if (correct/total>0.93 and correct/total>best_val_acc):
        best_val_acc=correct/total
                
        # torch.save(
        #     (net.state_dict(), optimizer.state_dict()), os.path.join("runs/kinetic-gan/classi_models", "checkpoint_ep{}_iterloss{}_valloss{}_valacc{}.pt".format(epoch,str(running_loss/epoch_steps),str(val_loss/val_steps),str(correct/total)))
        # )
        torch.save(
            (net.state_dict()), os.path.join(checkpoint_save_dir, "checkpoint_ep{}_iterloss{}_valloss{}_valacc{}.pt".format(epoch,str(running_loss/epoch_steps),str(val_loss/val_steps),str(correct/total)))
        )
    running_loss = 0.0
    
print("Finished epoch {}".format(epoch))




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
print("train_loss_values: ")
print(train_loss_values)
plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'train_loss_curve.png'))
plt.close()

print("val_acc_values: ")
print(val_acc_values)
plt.plot(val_acc_values, label='Val acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('Val Acc Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'val_acc_curve.png'))
plt.close()

print("val_loss_values: ")
print(val_loss_values)
plt.plot(val_loss_values, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Val Loss Curve')
plt.legend()
plt.savefig(os.path.join(graph_save_dir,'val_loss_curve.png'))
plt.close()