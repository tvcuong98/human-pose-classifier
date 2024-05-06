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
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--min_num_epochs", type=int, default=5, help="min number of epochs per trial, even when it perform bad -> also the grace period in AHSA scheduler")
parser.add_argument("--max_num_epochs", type=int, default=50, help="maximum number of epochs per trial, even when it perform well")
parser.add_argument("--num_samples", type=int, default=50, help="number of samples or configurations or trials will be conducted for EACH of the hyperparameter")
parser.add_argument("--model",type=str,help="The class name of the model")
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--latent",type=int,  nargs='+',default=[512, 256, 128, 64],help="the list of the hidden dims you want to test with")
parser.add_argument("--drop_out_p",type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],help="the list of the dropout you want to test with")
parser.add_argument("--lr", type=float, nargs="+",default=[0.0001, 0.1], help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=2, help="number of channels (coordinates)")
parser.add_argument("--train", type=str, help="path to data")
parser.add_argument("--test", type=str, help="path to data")
parser.add_argument("--val", type=str, help="path to data")
#parser.add_argument("--output_dir",type=str, default="output")
opt = parser.parse_args()
#python ray_torch_pose_classifier_full.py --min_num_epochs 5 --max_num_epochs 50 --num_samples 50 --model RobustPoseClassifier --batch_size 512 --latent 512 256 128 64 --drop_out_p 0.0 0.1 0.2 0.3 --lr 0.0001 0.1 --b1 0.5 --b2 0.999 --train /home/edabk-lab/cuong/human-pose-classifier/data/kp_16_cover_modes/uncover/trainuncover.csv --val /home/edabk-lab/cuong/human-pose-classifier/data/kp_16_cover_modes/uncover/testuncover.csv --test /home/edabk-lab/cuong/human-pose-classifier/data/kp_16_cover_modes/uncover/testuncover.csv

def train_pose_classifier(config):
    drop_out_p = config["drop_out_p"]
    latent = config["latent"]
    lr = config["lr"]
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
    optimizer = optim.Adam(net.parameters(), lr=lr,betas=(opt.b1,opt.b2))

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

    for epoch in range(opt.max_num_epochs):  # loop over the dataset multiple times
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
        y_true, y_pred = [], []  # For calculating other metrics
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

                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        # Calculate precision, recall, F1-score, and AUC
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        # auc = roc_auc_score(y_true, outputs.cpu(), multi_class='ovr')
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir: # the tempfile will be automatically deleted after this "with" ends
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir) # creates a checkpoint object from a local directory that contains checkpoint data
                                                                        # print(checkpoint.path) = /tmp/tmp7rahl3ff
            train.report(
                {"loss": (val_loss / val_steps), 
                 "accuracy": accuracy,
                 "precision": precision,
                 "recall": recall,
                 "f1":f1
                #  "auc":auc
                },
                checkpoint=checkpoint,
            )
    print("Finished Training")

def main(num_samples, max_num_epochs, gpus_per_trial, opt,smoke_test=False):
    print(list(opt.lr))
    config = {
        "lr": tune.loguniform(float(opt.lr[0]), float(opt.lr[1])),
        "drop_out_p": tune.grid_search(opt.drop_out_p),
        "latent": tune.grid_search(opt.latent)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs, # this is the maximum number of iterations a trial can run, even if it perform well
        grace_period=opt.min_num_epochs, # each trial is guaranteed to run for at least this much iterations
        reduction_factor=2) # how aggressively trials (different hyperparameter configurations) are eliminated
                            # when reduction_factor =2 , half of them are eliminated, 
                            # if =3 , two-third of trials are eliminated
    
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
    print("Best trial final validation precision: {}".format(
        best_result.metrics["precision"]))
    print("Best trial final validation recall: {}".format(
        best_result.metrics["recall"]))
    print("Best trial final validation f1-score: {}".format(
        best_result.metrics["f1-score"]))
    print("Best trial final validation auc: {}".format(
        best_result.metrics["auc"]))

    #test_best_model(best_result, smoke_test=smoke_test)
SMOKE_TEST = False
main(num_samples=50, max_num_epochs=50, gpus_per_trial=0.2, opt=opt,smoke_test=SMOKE_TEST)