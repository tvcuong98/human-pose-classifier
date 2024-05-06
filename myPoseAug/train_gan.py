import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataset import Ske_dataset
from generator import PoseGenerator
from discriminator import Pos2DDiscriminator
from utils import init_weights
from losses import get_adversarial_loss, get_diff_loss,get_classification_loss,get_feedback_loss
# Only discriminator will be used as input
# The generator are not used here, we assume that the data_fake have been created else where and just use it here
def main(args):
    if torch.cuda.is_available():  device = "cuda:0"
    start_epoch = 0

    ## Here comes the models
    generator = PoseGenerator(blr_tanhlimit=args.blr_tanhlimit, input_size=16 * 2) # only use the args.blr_tanhlimit
    discriminator = Pos2DDiscriminator(num_joints=16, kcs_channel=256, channel_mid=100)
    generator.to(device)
    discriminator.to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    print("==> Total G parameters: {:.2f}M".format(sum(p.numel() for p in generator.parameters()) / 1000000.0))
    print("==> Total D parameters: {:.2f}M".format(sum(p.numel() for p in discriminator.parameters()) / 1000000.0))
    ## for criterion 
    gan_criterion = nn.MSELoss(reduction='mean').to(device)
    classification_criterion = nn.CrossEntropyLoss().to(device)

    ## for optimizer
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=args.G_lr)
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.D_lr)


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
        "epoch_G_loss"    : [],
        "epoch_D_loss"    :[],
        "epoch_classi_train_loss:" : [],
        "epoch_classi_train_acc:" : [],
        "epoch_classi_val_loss:" : [],
        "epoch_classi_val_acc:" : []
    }

    for epoch in range(start_epoch,args.num_epochs):
        running_loss = 0.0
        epoch_steps = 0
        #### training generator and discriminator for GAN #####
        #### training generator 
        for i, data in enumerate(trainloader):
            data_real, labels = data
            data_real, labels = data_real.float().to(device), labels.type(torch.LongTensor).to(device)
            haah = discriminator(data_real)
            #### training generator 
            # G_optimizer.zero_grad()
            # data_fake_dict = generator(data_real)
            # data_fake = data_fake_dict['pose_ba']
            # _, generator_loss = get_adversarial_loss(discriminator,data_real,data_fake,gan_criterion)
            # generator_loss.backward()
            # G_optimizer.step()

            # ## training discriminator 
            # data_fake = data_fake_dict['pose_rt'].detach()  # Detach here
            # D_optimizer.zero_grad() 
            # # Recalculate adv_loss since the graph has been modified
            # adv_loss, _ = get_adversarial_loss(discriminator, data_real, data_fake, gan_criterion) 
            # adv_loss.backward()
            # D_optimizer.step()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--G_lr", type=float, default=0.0002, help="adam: lr for generator")
    parser.add_argument("--D_lr", type=float, default=0.0001, help="adam: lr for discriminator")
    parser.add_argument("--blr_tanhlimit", type=float, default=2e-1, help="bone length change limit")
    parser.add_argument("--train",type=str,help="path to the output folder")
    parser.add_argument("--test",type=str,help="path to the output folder of tensorboard")
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    args = get_args()
    main(args)





