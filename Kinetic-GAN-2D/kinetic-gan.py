import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from shutil import copyfile

from models.generator import Generator
from models.discriminator import Discriminator
from feeder.feeder import Feeder
from utils import general
from visualization.display import plot
import time
import torch
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--mlp_dim", type=int, default=4, help="mapping network depth")
parser.add_argument("--n_classes", type=int, default=60, help="number of classes for dataset")
parser.add_argument("--t_size", type=int, default=64, help="size of each temporal dimension")
parser.add_argument("--v_size", type=int, default=25, help="size of each spatial dimension (vertices)")
parser.add_argument("--channels", type=int, default=3, help="number of channels (coordinates)")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per generator's iteration")
parser.add_argument("--lambda_gp", type=int, default=10, help="Loss weight for gradient penalty in WGAN-GP Loss")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between action sampling")
parser.add_argument("--checkpoint_interval", type=int, default=10000, help="interval between model saving")
parser.add_argument("--G_pretrained",type=str,help="pretrained G model")
parser.add_argument("--D_pretrained",type=str,help="pretrained D model")
parser.add_argument("--dataset", type=str, default="ntu", help="dataset")
parser.add_argument("--csv_path", type=str, default="/Project/ske/data/swaptraingt16.csv", help="path to data")
parser.add_argument("--runs",type=str,help="path to the output folder")
parser.add_argument("--tb_runs",type=str,help="path to the output folder of tensorboard")
opt = parser.parse_args()
print(opt)

# Specify your desired log directory
log_dir = opt.tb_runs
writer = SummaryWriter(log_dir=log_dir) 



out        =  opt.runs
models_out  = os.path.join(out, 'models')
actions_out = os.path.join(out, 'actions')
png_samples_out = os.path.join(out, 'png_samples')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(actions_out): os.makedirs(actions_out)
if not os.path.exists(png_samples_out): os.makedirs(png_samples_out)


# Save config file and respective generator and discriminator for reproducibilty
config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

copyfile(os.path.basename(__file__), os.path.join(out, os.path.basename(__file__)))
copyfile('models/generator.py', os.path.join(out, 'generator.py'))
copyfile('models/discriminator.py', os.path.join(out, 'discriminator.py'))

cuda = True if torch.cuda.is_available() else False
print('CUDA',cuda)

# Models initialization
generator     = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.t_size, opt.mlp_dim, dataset=opt.dataset)
discriminator = Discriminator(opt.channels, opt.n_classes, opt.t_size, opt.latent_dim, dataset=opt.dataset)

if cuda:
    generator.cuda()
    discriminator.cuda()


# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataset=Feeder(opt.csv_path,opt.v_size,opt.t_size),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=opt.n_cpu
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


if (opt.G_pretrained!=None and opt.D_pretrained!=None):
    print("loading from -->")
    print(opt.G_pretrained)
    print(opt.D_pretrained)
    G_model_state, G_optimizer_state = torch.load(opt.G_pretrained)
    generator.load_state_dict(G_model_state)
    optimizer_G.load_state_dict(G_optimizer_state)
    D_model_state, D_optimizer_state = torch.load(opt.D_pretrained)
    discriminator.load_state_dict(D_model_state)
    optimizer_D.load_state_dict(D_optimizer_state)


Tensor     = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_action(n_row,epoch,batch):
    z = Variable(Tensor(np.random.normal(0, 1, (1*n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(1) for num in range(n_row)])
    labels = Variable(LongTensor(labels)) #tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device='cuda:0')
    print("sample_labels: ",labels)
    gen_imgs = generator(z, labels) #torch.Size([9, 2, 32, 16])
    print("gen_imgs_shape: ",gen_imgs.shape)
    print("gen_imgs.data[i] shape ",gen_imgs.data[1].shape)
    for i in range(9):
        plot(gen_imgs.data[i].cpu(),png_samples_out,f"epch_{epoch}_bt_{batch}_class_{i}")
        # try:
        #     #print(os.path.join(gif_samples_out,f"epch_{epoch}_bt_{batch}_class_{i}.gif"))
        #     plot(gen_imgs.data[i].cpu(),os.path.join(png_samples_out,f"epch_{epoch}_bt_{batch}_class_{i}"))
        # except:
        #     print("not good")
        #     continue
    # for i in range(9):
    #     try:
    #         additional_display_samples.plot(gen_imgs.data[i],str("epch{}_batch{}_class{}").format(epoch,batches_done,i),samples_out)
    #     except:
    #         continue
    # with open(os.path.join(actions_out, str(batches_done)+'.npy'), 'wb') as npf:
    #     np.save(npf, gen_imgs.data.cpu())

"""
The gradient_penalty represents a loss term added to the discriminator's loss in the WGAN-GP. 
The aim is to minimize this gradient_penalty. 
By doing so, the training process forces the discriminator to have gradients with norms closer to 1, 
encouraging smoother decision boundaries between real and fake data and ultimately improving the training stability of the WGAN.
"""


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    # It is like a random fusion ratio between a batch of real samples and a batch of fake samples
    # It is not really important whether it it real or it is fake right now, because all we care about in gradient penalty is :
    # The gradients of the discriminator output w.r.t a ARBITRARY, RANDOM sample, whether that sample is real or is fake
    # And we are going to penaltize if that gradients get too high, we want its norm close to 1
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True) 
    d_interpolates = D(interpolates, labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    # calculate gradients of the discriminator's output (d_interpolates) with respect to the interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake, #shape the resulting gradients. 
                            #Since d_interpolates has the same dimensions as fake (a tensor of ones), 
                            #the calculated gradients will also have the same dimensions.
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0] # this have shape [m,2,1,16], because the fake have shape [m,2,1,16]
    gradients = gradients.reshape(gradients.size(0), -1) #  [m,2,1,16] - > [m,32]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() #And we are going to penaltize if that gradients get too high, 
                                                                    #we want its norm close to 1
    return gradient_penalty # a scalar number


# ----------
#  Training
# ----------

loss_d, loss_g = [], []
batches_done   = 0
for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, (imgs, labels) in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure input
        imgs = imgs[:,:,:opt.t_size,:]
        real_imgs = Variable(imgs.type(Tensor))
        labels    = Variable(labels.type(LongTensor))
        #plot_gif(real_imgs[200].cpu(),"haha.gif")
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of actions
        fake_imgs = generator(z, labels)

        # Real actions
        real_validity = discriminator(real_imgs, labels)
        # Fake actions
        fake_validity = discriminator(fake_imgs, labels)

        
        #print(fake_imgs.data.shape) # torch.Size([128, 2, 1, 16])
        #print(real_imgs.data.shape) # torch.Size([128, 2, 1, 16])
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
        # d_loss = -torch.mean(real_validity**2) + torch.mean(fake_validity**2) + opt.lambda_gp * gradient_penalty
        # Explain : -torch.mean(real_validity) -> "negative" because we want to maximize it instead of minimizing it
        #                                      -> when the sample is real, the real_validity should be "high" ~ "positive number"
        #                                      -> so we should maximizing it so that it more positive = more "realness"
        # Explain : torch.mean(fake_validity) -> "positive" because we want to minimize it , like loss function usually do 
        #                                      -> when the sample is real, the fake_validity should be "low" ~ "negative number"
        #                                      -> so we should minimizing it so that it more negative = more "fakeness"
         
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator after n_critic discriminator steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of actions
            fake_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake actions
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()


    loss_d.append(d_loss.data.cpu())
    loss_g.append(g_loss.data.cpu())
    if epoch % opt.sample_interval == 0:
        sample_action(n_row=opt.n_classes,epoch=epoch,batch=batches_done)
        
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save((generator.state_dict(),optimizer_G.state_dict()), os.path.join(models_out, "generator_ep_%d_bt_%d.pth" % (epoch,batches_done)))
        torch.save((discriminator.state_dict(),optimizer_D.state_dict()), os.path.join(models_out, "discriminator_ep_%d_bt_%d.pth" % (epoch,batches_done)))
    writer.add_scalar('Loss/generator', g_loss, epoch)
    writer.add_scalar('Loss/discriminator', d_loss, epoch)
    writer.add_scalars('Loss/gen_dis', {
                      'Generator': g_loss,
                      'Discriminator': d_loss
                     }, epoch)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time : %.2f sec]"
        % (epoch, opt.n_epochs, i, len(dataloader) , d_loss.item(), g_loss.item(),epoch_time)
    )
loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('kinetic-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')