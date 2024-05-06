import torch
from torch.autograd import Variable
def get_adversarial_loss(discriminator,data_real,data_fake,criterion,device=torch.device("cuda")):
    real_validity = discriminator(data_real)
    fake_validity = discriminator(data_fake)

    real_label = Variable(torch.ones(real_validity.size())).to(device)
    fake_label = Variable(torch.zeros(fake_validity.size())).to(device)

    adversarial_real_loss = criterion(real_validity,real_label)
