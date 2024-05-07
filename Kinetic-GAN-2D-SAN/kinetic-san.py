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
import time
from models.generator import Generator
from models.discriminator import Discriminator
from feeder.feeder import Feeder
from utils import general
from visualization.display import plot

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
cuda = True if torch.cuda.is_available() else False
# datatypes:
Tensor     = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_action(n_row,epoch,batch,png_samples_out,generator):
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
    d_interpolates = D(interpolates, labels,False)
    fake = Variable(Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
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

def update_discriminator(imgs_real, class_ids, discriminator, generator, optimizer, opt):
    bs = imgs_real.size(0)
    device = imgs_real.device

    optimizer.zero_grad()

    # for data (ground-truth) distribution
    disc_real = discriminator(imgs_real, class_ids, flg_train=True)
    loss_real = eval('compute_loss_'+opt.model)(disc_real, loss_type='real')

    # for generator distribution
    latent = torch.randn(bs, opt.latent_dim, device=device)
    imgs_fake = generator(latent, class_ids)
    disc_fake = discriminator(imgs_fake.detach(), class_ids, flg_train=True)
    loss_fake = eval('compute_loss_'+opt.model)(disc_fake, loss_type='fake')

    # compute gradient penalty:
    gradient_penalty = compute_gradient_penalty(discriminator, imgs_real, imgs_fake, class_ids)
    
    loss_d = loss_real + loss_fake + opt.lambda_gp * gradient_penalty
    loss_d.backward()
    optimizer.step()
    return loss_d


def update_generator(num_class, discriminator, generator, optimizer, opt, device):
    optimizer.zero_grad()

    bs = opt.batch_size
    latent = torch.randn(bs, opt.latent_dim, device=device)

    class_ids = torch.randint(num_class, size=(bs,), device=device)
    batch_fake = generator(latent, class_ids)

    disc_gen = discriminator(batch_fake, class_ids, flg_train=False)
    loss_g = - disc_gen.mean()
    loss_g.backward()
    optimizer.step()

    return loss_g


def compute_loss_gan(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss = (1. - disc).relu().mean() # Hinge loss
    else: # 'fake' == loss_type
        loss = (1. + disc).relu().mean() # Hinge loss

    return loss


def compute_loss_san(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss_fun = (1. - disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = - disc['dir'].mean() # Wasserstein loss for omega
    else: # 'fake' == loss_type
        loss_fun = (1. + disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = disc['dir'].mean() # Wasserstein loss for omega
    loss = loss_fun + loss_dir

    return loss


# def save_images(imgs, idx, dirname='test'):
#     import numpy as np
#     if imgs.shape[1] == 1:
#         imgs = np.repeat(imgs, 3, axis=1)
#     fig = plt.figure(figsize=(10, 10))
#     gs = gridspec.GridSpec(10, 10)
#     gs.update(wspace=0.05, hspace=0.05)
#     for i, sample in enumerate(imgs):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.transpose((1,2,0)))

#     if not os.path.exists('out/{}/'.format(dirname)):
#         os.makedirs('out/{}/'.format(dirname))
#     plt.savefig('out/{0}/{1}.png'.format(dirname, str(idx).zfill(3)), bbox_inches="tight")
#     plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1, help="gpu device to use")
    parser.add_argument("--model", type=str, default="san", help="model to use")
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
    return opt


def main(opt):
    print("Arguments: ")
    print(opt)
    print("------------------------------------------------")
    if cuda:
        device = f'cuda:{opt.device}' if opt.device is not None else 'cpu'
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
    

    # dataloading
    # Configure data loader and dataset
    dataset = Feeder(opt.csv_path,opt.v_size,opt.t_size)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True, 
        persistent_workers=True,
        num_workers=opt.n_cpu
    )
    n_iters_per_epoch = len(dataloader)
    print(f"the length of dataset is {len(dataset)}, with {n_iters_per_epoch} iteration and batch size of {opt.batch_size}")
    # model
    # Models initialization
    generator     = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.t_size, opt.mlp_dim, dataset=opt.dataset)
    discriminator = Discriminator(opt.channels, opt.n_classes, opt.t_size, opt.latent_dim, dataset=opt.dataset)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in generator.parameters()) / 1000000.0))
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in discriminator.parameters()) / 1000000.0))
    # # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # Optimizers RMSprop
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, alpha=0.99)  
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, alpha=0.99)
    # loading pretrained if specified
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

    # # eval initial states
    # num_samples_per_class = 10
    # with torch.no_grad():
    #     latent = torch.randn(num_samples_per_class * num_class, params["dim_latent"]).cuda()
    #     class_ids = torch.arange(num_class, dtype=torch.long,
    #                              device=device).repeat_interleave(num_samples_per_class)
    #     imgs_fake = generator(latent, class_ids)

    # main training loop
    loss_ds, loss_gs = [], []
    batches_done   = 0


    for epoch in range(opt.n_epochs):
        start_time = time.time()
        for i, (imgs, labels) in enumerate(dataloader):
            batches_done = epoch * len(dataloader) + i
            # Configure input
            imgs = imgs[:,:,:opt.t_size,:]
            real_imgs = Variable(imgs.type(Tensor)).to(device)
            labels    = Variable(labels.type(LongTensor)).to(device)
            loss_d = update_discriminator(real_imgs, labels, discriminator, generator, optimizer_D, opt)
            if i % opt.n_critic == 0:
                loss_g = update_generator(opt.n_classes, discriminator, generator, optimizer_G, opt, device)


        loss_ds.append(loss_d.data.cpu())
        loss_gs.append(loss_g.data.cpu())

        if epoch % opt.sample_interval ==0:
            sample_action(n_row=opt.n_classes,epoch=epoch,batch=batches_done,png_samples_out=png_samples_out,generator=generator)
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save((generator.state_dict(),optimizer_G.state_dict()), os.path.join(models_out, "generator_ep-%d_bth-%d.pth" % (epoch,batches_done)))
            torch.save((discriminator.state_dict(),optimizer_D.state_dict()), os.path.join(models_out, "discriminator_ep-%d_bth-%d.pth" % (epoch,batches_done)))
        writer.add_scalar('Loss/generator', loss_g, epoch)
        writer.add_scalar('Loss/discriminator', loss_d, epoch)
        writer.add_scalars('Loss/gen_dis', {'Generator':loss_g,'Discriminator': loss_d}, epoch)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time : %.2f sec]"
            % (epoch, opt.n_epochs, i, n_iters_per_epoch , loss_d.item(), loss_g.item(),epoch_time)
        )
    #     # eval
    #     with torch.no_grad():
    #         latent = torch.randn(num_samples_per_class * num_class, params["dim_latent"]).cuda()
    #         class_ids = torch.arange(num_class, dtype=torch.long,
    #                                  device=device).repeat_interleave(num_samples_per_class)
    #         imgs_fake = generator(latent, class_ids).cpu().data.numpy()
    #         save_images(imgs_fake, n, dirname=experiment_name)
    
    # torch.save(generator.state_dict(), ckpt_dir + "generator.pt")
    # torch.save(discriminator.state_dict(), ckpt_dir + "discriminator.pt")


if __name__ == '__main__':
    opt = get_args()
    main(opt)
