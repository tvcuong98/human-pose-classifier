import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from fcn_classifier import HeavyPoseClassifier
from dataset import Ske_dataset
from generator import PoseGenerator
from discriminator import Pos2DDiscriminator
from utils import init_weights,set_grad,has_nan
from losses import get_adversarial_loss, get_diff_loss,get_classification_loss,get_feedback_loss
from visualization import plot
import random
import os

############################ SAMPLE USAGE :
#python train.py --runs cover2 --epochs 500 --batch_size 64 --train /home/edabk/cuong/human-pose-classifier/data/kp_16_cover_modes/cover2/traincover2.csv --test /home/edabk/cuong/human-pose-classifier/data/kp_16_cover_modes/cover2/testcover2.csv
############################ GOOD LUCK
# Only discriminator will be used as input
# The generator are not used here, we assume that the data_fake have been created else where and just use it here
def main(args):
    # make the output files
    output_dir        =  args.output_dir
    runs              =  args.runs
    runs_dir          =  os.path.join(output_dir,runs)
    if not os.path.exists(runs_dir): os.makedirs(runs_dir)


    if torch.cuda.is_available():  device = "cuda:0"
    start_epoch = 0

    ## Here comes the models
    classifier = HeavyPoseClassifier()
    generator = PoseGenerator(blr_tanhlimit=args.blr_tanhlimit, input_size=16 * 2,num_stage_BA=4,num_stage_BL=4,num_stage_RT=4) # only use the args.blr_tanhlimit
    discriminator = Pos2DDiscriminator(num_joints=16, kcs_channel=512, channel_mid=100)
    classifier.to(device)
    generator.to(device)
    discriminator.to(device)
    # classifier.apply(init_weights)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    print("==> Total G parameters: {:.2f}M".format(sum(p.numel() for p in generator.parameters()) / 1000000.0))
    print("==> Total D parameters: {:.2f}M".format(sum(p.numel() for p in discriminator.parameters()) / 1000000.0))
    print("==> Total C parameters: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1000000.0))
    num_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in classifier.parameters())
    print("Trainable param : {} total param : {}".format(num_trainable_params,num_params))
    ## for criterion 
    gan_criterion = nn.MSELoss(reduction='mean').to(device)
    classification_criterion = nn.CrossEntropyLoss()

    ## for optimizer
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=args.G_lr)
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.D_lr)
    C_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.C_lr)

    # create a dataset here
    trainset = Ske_dataset(csv_path=args.train)
    testset = Ske_dataset(csv_path=args.test)
    # create a dataloader here, named trainloader
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    # create a dataloader here for testing, named testloader
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    logger = {
        "epoch_best_val_acc" : 0.0,
        "best_epoch"         : 0,
        "epoch_G_loss"    : [],
        "epoch_D_loss"    :[],
        "epoch_classi_train_loss:" : [],
        "epoch_classi_train_acc:" : [],
        "epoch_classi_val_loss:" : [],
        "epoch_classi_val_acc:" : []
    }

    for epoch in range(start_epoch,args.epochs):
        running_loss = 0.0
        epoch_steps = 0
        #### training generator and discriminator for GAN #####
        #### training generator 
        for i, data in enumerate(trainloader):
            data_real, labels = data
            data_real, labels = data_real.float().to(device), labels.type(torch.LongTensor).to(device)
            ##################################################
            #######      Train Generator     #################
            ##################################################
            if (epoch >= args.start_schedule[0] and i % args.update_schedule[0]==0):
                set_grad([classifier],False)
                set_grad([discriminator],False)
                set_grad([generator],True)
                G_optimizer.zero_grad()
                data_fake_dict = generator(data_real)
                data_fake = data_fake_dict['pose_bl']
                adv_loss, _ = get_adversarial_loss(discriminator,data_real,data_fake,gan_criterion)
                feedback_loss = get_feedback_loss(args,classifier,data_real,data_fake_dict,labels,epoch)
                diff_loss = get_diff_loss(args, data_fake_dict)
                if epoch > args.start_schedule[2]: # but this always happen because we are setting it to epoch 0
                    G_loss = adv_loss * args.gloss_factor_adv + \
                            feedback_loss * args.gloss_factor_feedback + \
                            diff_loss * args.gloss_factor_diff
                else: 
                    G_loss = adv_loss * args.gloss_factor_adv + \
                            diff_loss * args.gloss_factor_diff
                G_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)
                G_optimizer.step()

            ##################################################
            #######      Train Discriminator     #############
            ##################################################
            if (epoch >= args.start_schedule[1] and i % args.update_schedule[1]==0):
                set_grad([classifier],False)
                set_grad([discriminator],True)
                set_grad([generator],False)
                data_fake = data_fake_dict['pose_bl'].detach()  # Detach here
                D_optimizer.zero_grad() 
                # Recalculate adv_loss since the graph has been modified
                adv_loss, _ = get_adversarial_loss(discriminator, data_real, data_fake, gan_criterion) 
                D_loss = adv_loss
                D_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)
                D_optimizer.step()
            ##################################################
            #######      Train Classifier     #############
            ##################################################
            if (epoch >= args.start_schedule[2] and i % args.update_schedule[2]==0):
                set_grad([classifier],True)
                set_grad([discriminator],False)
                set_grad([generator],False)
                C_optimizer.zero_grad()
                if (epoch < args.start_schedule[0]):
                    real_classi_loss = get_classification_loss(data_real,classifier,labels,classification_criterion)
                    classi_loss = real_classi_loss
                else:
                    data_fake = data_fake_dict['pose_bl'].detach()  # Detach here
                    fake_classi_loss = get_classification_loss(data_fake,classifier,labels,classification_criterion)
                    real_classi_loss = get_classification_loss(data_real,classifier,labels,classification_criterion)
                    classi_loss = real_classi_loss + fake_classi_loss
                classi_loss.backward()
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1)
                C_optimizer.step()
                # print statistics
                running_loss += classi_loss.item()
                epoch_steps += 1
        ##################################################
        #######      Validation Classifier     #############
        ##################################################
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.type(torch.LongTensor).to(device)
                outputs = classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = classification_criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        if (logger["epoch_best_val_acc"] <  correct/total): 
            logger["epoch_best_val_acc"] = correct/total
            logger["best_epoch"] = epoch
            torch.save((classifier.state_dict()),os.path.join(runs_dir,"best_ckpt.pt"))
        rand_idx = random.randint(0, len(data_real)-1) # since both inclusive
        if (epoch >= args.start_schedule[0]) :# starting to have data_fake
            plot(data_fake[rand_idx],runs_dir,"fake")
            plot(data_real[rand_idx],runs_dir,"real")
            print(f"epoch_{epoch}_ganloss_{G_loss}_disloss_{D_loss}_classi_loss_{classi_loss}_val_loss_{val_loss/val_steps}_acc_{correct/total}")
        else:
            plot(data_real[rand_idx],runs_dir,"real")
            print(f"epoch_{epoch}_classi_loss_{classi_loss}_val_loss_{val_loss/val_steps}_acc_{correct/total}")

        
    os.rename(os.path.join(runs_dir,"best_ckpt.pt"), os.path.join(runs_dir,f"best_ep_{logger['best_epoch']}_acc_{logger['epoch_best_val_acc']}.pt"))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str,default="./poseaug_output",help="folder for outputing result")
    parser.add_argument("--runs",type=str,help="name each runs, for example for diffrent data. Output will be stored in <output_dir>/<runs>/")
    parser.add_argument("--start_schedule",type=int,  nargs='+',default=[5, 5, 0],help="generator,discriminator,classifier start training at their corresponding epoch")
    parser.add_argument("--update_schedule",type=int,  nargs='+',default=[1, 2, 1],help="generator,discriminator,classifier get trained every .. epochs. The first number are for generator, so on")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--G_lr", type=float, default=0.001, help="adam: lr for generator")
    parser.add_argument("--D_lr", type=float, default=0.001, help="adam: lr for discriminator")
    parser.add_argument("--C_lr", type=float, default=0.002, help="adam: lr for classifier")
    parser.add_argument('--blr_tanhlimit', default=2e-1, type=float, help='bone length change limit.')
    parser.add_argument('--blr_limit', default=1e-1, type=float, help='bone length change limit.')
    parser.add_argument("--train",type=str,help="path to data train")
    parser.add_argument("--test",type=str,help="path to data test")
    parser.add_argument('--ba_range_m', default=15.5e-2, type=float, help='bone angle modification range.')
    parser.add_argument('--ba_range_w', default=14.5e-2, type=float, help='bone angle modification range.')
    parser.add_argument("--hardratio_ba_s",type=float,default=3,help="starting value for hardratio ba")
    parser.add_argument("--hardratio_ba",type=float,default=5,help="ending value for hardratio ba")
    parser.add_argument("--hardratio_std_ba",type=float,default=2,help="standard deviation for hardratio ba")
    parser.add_argument("--gloss_factordiv_ba",type=float,default=0.,help="factor for range difference loss")
    parser.add_argument('--gloss_factorfeedback_ba', default=1e-1, type=float, help='factor for feedback loss from ba.')
    parser.add_argument("--hardratio_bl_s",type=float,default=3,help="starting value for hardratio rt")
    parser.add_argument("--hardratio_bl",type=float,default=5,help="ending value for hardratio rt")
    parser.add_argument("--hardratio_std_bl",type=float,default=2,help="standard deviation for hardratio rt")
    parser.add_argument('--gloss_factordiv_bl', default=0., type=float, help='factor for range difference loss')
    parser.add_argument('--gloss_factorfeedback_bl', default=1e-1, type=float, help='factor for feedback loss from bl.')
    parser.add_argument('--gloss_factor_adv',default=6,type=float,help="factor for adversarial loss in gen loss function")
    parser.add_argument('--gloss_factor_diff',default=3,type=float,help="factor for diff loss in gen loss function")
    parser.add_argument('--gloss_factor_feedback',default=1,type=float,help="factor for feedback loss in gen loss function")
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    args = get_args()
    main(args)





