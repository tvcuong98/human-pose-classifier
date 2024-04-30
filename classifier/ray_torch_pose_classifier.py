import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from models import HeavyPoseClassifier
from PoseClassifier import RobustPoseClassifier
from dataset import Dataset_ske,Robust_Dataset_ske
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--model",type=str,help="The class name of the model")
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--hidden_dims",type=list, default=[512,256,128,64],help="the list of the hidden dims you want to test with")
parser.add_argument("--drop_out_p",type=list, default=[0.0,0.1,0.2,0.3],help="the list of the dropout you want to test with")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=2, help="number of channels (coordinates)")
parser.add_argument("--checkpoint_interval", type=int, default=6000, help="interval between model saving")
parser.add_argument("--pretrained_model",type=str,help="training from checkpoint")
parser.add_argument("--train", type=str, help="path to data")
parser.add_argument("--test", type=str, help="path to data")
parser.add_argument("--val", type=str, help="path to data")
#parser.add_argument("--output_dir",type=str, default="output")
opt = parser.parse_args()
def train_pose_classifier(config):
    drop_out_p = config["drop_out_p"]
    latent = config["latent"]
    if (opt.model == "RobustPoseClassifier"): 
        net = RobustPoseClassifier(in_channels=2, n_classes=9, t_size=1, latent=latent)
        trainset=Robust_Dataset_ske(opt.train)
        valset=Robust_Dataset_ske(opt.test)
        #testset=Robust_Dataset_ske(opt.test)
    elif (opt.model == "HeavyPoseClassifier"):
        net = HeavyPoseClassifier(input_size=32, num_classes=9, drop_out_p=drop_out_p,hidden_dims=latent)
        trainset=Dataset_ske(opt.train)
        valset=Dataset_ske(opt.test)
        #testset=Dataset_ske(opt.test)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )
    valloader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    for epoch in range(opt.n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

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

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir: # the tempfile will be automatically deleted after this "with" ends
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir) # creates a checkpoint object from a local directory that contains checkpoint data
                                                                        # print(checkpoint.path) = /tmp/tmp7rahl3ff
            
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

def main(num_samples, max_num_epochs, gpus_per_trial, opt,smoke_test=False):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "drop_out_p": tune.grid_search([0.0,0.2,0.4,0.6]),
        "latent": tune.grid_search([512,256,128,64])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_pose_classifier),
            resources={"cpu": 0, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    #test_best_model(best_result, smoke_test=smoke_test)
SMOKE_TEST = False
main(num_samples=50, max_num_epochs=50, gpus_per_trial=0.2, opt=opt,smoke_test=SMOKE_TEST)