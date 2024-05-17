import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from classi_models.fcn_classifier import HeavyPoseClassifier
from classi_models.gcn_classifier import RobustPoseClassifier
from utils import init_weights,set_grad,has_nan
from losses import get_adversarial_loss, get_diff_loss,get_classification_loss,get_feedback_loss
from visualization import plot
import random
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter


#### Additional import from the discriminator and the generator of kinetic-gan
from models.generator import Generator
from models.discriminator import Discriminator
from feeder.feeder import Feeder
from utils import general
from visualization.display import plot
import time

############################ SAMPLE USAGE :
#CUDA_VISIBLE_DEVICES=0 python train.py --model fcn --hidden_dims 64 --epochs 1000 --batch_size 64 --runs --train --test 
#CUDA_VISIBLE_DEVICES=0 python train.py --model gcn --hidden_dims 512 --epochs 1000 --batch_size 64 --n_cpu 8 --latent_dim 512 --mlp_dim 8 --t_size 1 --v_size 16 --channels 2 --lambda_gp 10 --dataset h36m --G_pretrained --D_pretrained --runs --train --test
########################## GOOD LUCK
# Only discriminator will be used as input
# The generator are not used here, we assume that the data_fake have been created else where and just use it here
def main(args):
    Tensor     = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    # make the output files
    output_dir        =  args.output_dir
    runs              =  args.runs
    runs_dir          =  os.path.join(output_dir,runs)
    tb_runs_dir       =  os.path.join(runs_dir,args.tb_runs)
    if not os.path.exists(runs_dir): os.makedirs(runs_dir)
    if not os.path.exists(tb_runs_dir): os.makedirs(tb_runs_dir)
    log_dir = tb_runs_dir
    writer = SummaryWriter(log_dir=log_dir) 

    if torch.cuda.is_available():  device = "cuda:0"
    start_epoch = 0

    ## Here comes the models
    if args.model =="fcn":
        classifier = HeavyPoseClassifier(hidden_dims=args.hidden_dims)
    elif args.model =="gcn":
        classifier = RobustPoseClassifier(in_channels=2, n_classes=9, t_size=1, latent=args.hidden_dims)
    generator = Generator(args.latent_dim, args.channels, args.n_classes, args.t_size, args.mlp_dim, dataset=args.dataset)
    discriminator = Discriminator(args.channels, args.n_classes, args.t_size, args.latent_dim, dataset=args.dataset)
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


    ## load pretrained G and D models:
    if (args.G_pretrained!=None and args.D_pretrained!=None):
        print("loading from -->")
        print(args.G_pretrained)
        print(args.D_pretrained)
        G_model_state, G_optimizer_state = torch.load(args.G_pretrained)
        generator.load_state_dict(G_model_state)
        G_optimizer.load_state_dict(G_optimizer_state)
        D_model_state, D_optimizer_state = torch.load(args.D_pretrained)
        discriminator.load_state_dict(D_model_state)
        D_optimizer.load_state_dict(D_optimizer_state)


    # create a dataset here
    # trainset = Ske_dataset(csv_path=args.train)
    # testset = Ske_dataset(csv_path=args.test)
    # create a dataloader here, named trainloader
    trainloader = torch.utils.data.DataLoader(
        dataset=Feeder(args.train,args.v_size,args.t_size),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.n_cpu
    )
    # create a dataloader here for testing, named testloader
    testloader = torch.utils.data.DataLoader(
        dataset=Feeder(args.test,args.v_size,args.t_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )
    logger = {
        "epoch_best_val_acc" : 0.0,
        "best_epoch"         : 0,
        "best_model"      : None,
        "epoch_G_loss"    : [],
        "epoch_D_loss"    :[],
        "epoch_classi_train_loss:" : [],
        "epoch_classi_train_acc:" : [],
        "epoch_classi_val_loss:" : [],
        "epoch_classi_val_acc:" : []
    }

    for epoch in range(start_epoch,args.epochs):
        start_time = time.time()
        running_loss = {"adv_loss": 0.0,
                        "feedback_loss": 0.0,
                        "G_loss": 0.0,
                        "D_loss": 0.0,
                        "C_loss": 0.0
                        }
        epoch_steps = {"generator_steps":0,
                       "discriminator_steps":0,
                       "classifier_steps":0}
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

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
                data_fake = generator(z,labels)
                adv_loss, _ = get_adversarial_loss(discriminator,data_real,data_fake,gan_criterion)
                feedback_loss = get_feedback_loss(args,classifier,data_real,data_fake,labels,epoch)
                if epoch > args.start_schedule[2]: # but this always happen because we are setting it to epoch 0
                    G_loss = adv_loss * args.gloss_factor_adv + \
                            feedback_loss * args.gloss_factor_feedback
                else: 
                    G_loss = adv_loss * args.gloss_factor_adv
                G_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)
                G_optimizer.step()

                # This is the part where we sum the loss up for logging
                running_loss["adv_loss"] += adv_loss.item()
                running_loss["feedback_loss"] += feedback_loss.item()
                running_loss["G_loss"] += G_loss.item()
                epoch_steps["generator_steps"] += 1



            ##################################################
            #######      Train Discriminator     #############
            ##################################################
            if (epoch >= args.start_schedule[1] and i % args.update_schedule[1]==0):
                set_grad([classifier],False)
                set_grad([discriminator],True)
                set_grad([generator],False)
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
                data_fake = generator(z,labels)
                D_optimizer.zero_grad() 
                # Recalculate adv_loss since the graph has been modified
                adv_loss, _ = get_adversarial_loss(discriminator, data_real, data_fake, gan_criterion) 
                D_loss = adv_loss
                D_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)
                D_optimizer.step()
                # This is the part where we sum the loss up for logging
                running_loss["D_loss"] += D_loss.item()
                epoch_steps["discriminator_steps"] += 1

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
                    # Sample noise as generator input
                    z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
                    data_fake = generator(z,labels)
                    fake_classi_loss = get_classification_loss(data_fake,classifier,labels,classification_criterion)
                    real_classi_loss = get_classification_loss(data_real,classifier,labels,classification_criterion)
                    classi_loss = real_classi_loss + fake_classi_loss
                classi_loss.backward()
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1)
                C_optimizer.step()
                # This is the part where we sum the loss up for logging
                running_loss["C_loss"] += classi_loss.item()
                epoch_steps["classifier_steps"] += 1
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
        ##################################################
        #######      Logging after each epoc    #############
        ##################################################


        ##################################################
        #######      Logging in Tensorboard    #############
        ##################################################
        epoch_C_loss = running_loss["C_loss"]/epoch_steps["classifier_steps"]
        epoch_val_C_loss = val_loss / val_steps
        epoch_accuracy = correct/total
        writer.add_scalars('Losses', {
                        'C_loss': epoch_C_loss,
                        'val_C_loss': epoch_val_C_loss,
                        'val_C_acc' : epoch_accuracy
                        }, epoch)
        writer.add_scalar('Losses/C_loss', epoch_C_loss,epoch)
        writer.add_scalar('Losses/val_C_loss', epoch_val_C_loss,epoch)
        writer.add_scalar('Losses/val_C_acc', epoch_accuracy,epoch)
        if (epoch >= args.start_schedule[0]) :# starting to have data_fake, the gen and dis start training    
            epoch_adv_loss = running_loss["adv_loss"]/epoch_steps["generator_steps"]
            epoch_feedback_loss = running_loss["feedback_loss"]/epoch_steps["generator_steps"]
            epoch_G_loss = running_loss["G_loss"]/epoch_steps["generator_steps"]
            epoch_D_loss = running_loss["D_loss"]/epoch_steps["discriminator_steps"]
            writer.add_scalars('Losses', {
                        'G_adv_loss': epoch_adv_loss,
                        'G_fb_loss': epoch_feedback_loss,
                        'G_loss': epoch_G_loss,
                        'D_loss': epoch_D_loss,
                        }, epoch)
            writer.add_scalar('Losses/G_adv_loss', epoch_adv_loss,epoch)
            writer.add_scalar('Losses/G_fb_loss', epoch_feedback_loss,epoch)
            writer.add_scalar('Losses/G_loss', epoch_G_loss,epoch)
            writer.add_scalar('Losses/D_loss', epoch_D_loss,epoch)
######################
        #######      Logging for saving best, printing    #############
        ##################################################   
        if (logger["epoch_best_val_acc"] <  correct/total): 
            logger["epoch_best_val_acc"] = correct/total
            logger["best_epoch"] = epoch
            logger["best_model"] = classifier
        end_time = time.time()
        epoch_time = end_time - start_time
        """ We wont plot anything just yet

        rand_idx = random.randint(0, len(data_real)-1) # since both inclusive
        if (epoch >= args.start_schedule[0]) :# starting to have data_fake
            plot(data_fake[rand_idx],runs_dir,"fake")
            plot(data_real[rand_idx],runs_dir,"real")
            print(
                "[Epoch %d/%d] [D loss: %.4f] [G_adv loss: %.4f] [G_fb loss: %.4f] [G_diff loss: %.4f] [G loss: %.4f] [C loss: %.4f] [C vloss: %.4f] [Acc: %.4f] [Time : %.2f sec]"
                % (epoch, args.epochs, epoch_D_loss, epoch_adv_loss,epoch_feedback_loss,epoch_diff_loss,epoch_G_loss,epoch_C_loss,epoch_val_C_loss,epoch_accuracy,epoch_time)
            )
        else:
            plot(data_real[rand_idx],runs_dir,"real")
            print(
                "[Epoch %d/%d] [C loss: %.4f] [C vloss: %.4f] [Acc: %.4f] [Time : %.2f sec]"
                % (epoch,args.epochs,epoch_C_loss,epoch_val_C_loss,epoch_accuracy,epoch_time)
            )
        """
    best_model_save_file = f"best_ep_{logger['best_epoch']}_acc_{logger['epoch_best_val_acc']}.pt"
    torch.save((logger["best_model"].state_dict()),os.path.join(runs_dir,best_model_save_file))
    print(logger)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims",type=int,help="the number of hidden nodes, it depends on the model too")
    parser.add_argument("--model",type=str,help="either gcn or fcn")
    parser.add_argument("--output_dir",type=str,default="./ganaug_output",help="folder for outputing result")
    parser.add_argument("--runs",type=str,help="name each runs, for example for diffrent data. Output will be stored in <output_dir>/<runs>/")
    parser.add_argument("--tb_runs",type=str,default="tensorboard",help="the tensorboard directory, located inside the <runs> folder")
    parser.add_argument("--start_schedule",type=int,  nargs='+',default=[1, 1, 0],help="generator,discriminator,classifier start training at their corresponding epoch")
    parser.add_argument("--update_schedule",type=int,  nargs='+',default=[1, 1, 1],help="generator,discriminator,classifier get trained every .. epochs. The first number are for generator, so on")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--G_lr", type=float, default=0.001, help="adam: lr for generator")
    parser.add_argument("--D_lr", type=float, default=0.001, help="adam: lr for discriminator")
    parser.add_argument("--C_lr", type=float, default=0.002, help="adam: lr for classifier")
    parser.add_argument('--blr_tanhlimit', default=3e-1, type=float, help='bone length change limit.')
    parser.add_argument('--blr_limit', default=2e-1, type=float, help='bone length change limit.')
    parser.add_argument("--train",type=str,help="path to data train")
    parser.add_argument("--test",type=str,help="path to data test")
    parser.add_argument('--ba_range_m', default=20.5e-2, type=float, help='bone angle modification range.')
    parser.add_argument('--ba_range_w', default=16.5e-2, type=float, help='bone angle modification range.')
    parser.add_argument("--hardratio_ba_s",type=float,default=4,help="starting value for hardratio ba")
    parser.add_argument("--hardratio_ba",type=float,default=6,help="ending value for hardratio ba")
    parser.add_argument("--hardratio_std_ba",type=float,default=2,help="standard deviation for hardratio ba")
    parser.add_argument("--gloss_factordiv_ba",type=float,default=0.,help="factor for range difference loss")
    parser.add_argument('--gloss_factorfeedback_ba', default=1e-1, type=float, help='factor for feedback loss from ba.')
    parser.add_argument("--hardratio_bl_s",type=float,default=4,help="starting value for hardratio rt")
    parser.add_argument("--hardratio_bl",type=float,default=6,help="ending value for hardratio rt")
    parser.add_argument("--hardratio_std_bl",type=float,default=2,help="standard deviation for hardratio rt")
    parser.add_argument('--gloss_factordiv_bl', default=0., type=float, help='factor for range difference loss')
    parser.add_argument('--gloss_factorfeedback_bl', default=1e-1, type=float, help='factor for feedback loss from bl.')
    parser.add_argument('--gloss_factor_adv',default=6,type=float,help="factor for adversarial loss in gen loss function")
    parser.add_argument('--gloss_factor_diff',default=2,type=float,help="factor for diff loss in gen loss function")
    parser.add_argument('--gloss_factor_feedback',default=2,type=float,help="factor for feedback loss in gen loss function")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--mlp_dim", type=int, default=4, help="mapping network depth")
    parser.add_argument("--t_size", type=int, default=64, help="size of each temporal dimension")
    parser.add_argument("--v_size", type=int, default=25, help="size of each spatial dimension (vertices)")
    parser.add_argument("--channels", type=int, default=2, help="number of channels (coordinates)")
    parser.add_argument("--lambda_gp", type=int, default=10, help="Loss weight for gradient penalty in WGAN-GP Loss")
    parser.add_argument("--G_pretrained",type=str,help="pretrained G model")
    parser.add_argument("--D_pretrained",type=str,help="pretrained D model")
    parser.add_argument("--dataset", type=str, default="ntu", help="dataset")
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    args = get_args()
    main(args)





