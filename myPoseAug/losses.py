import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import has_nan,has_zero
### basic loss functions:
def rectifiedL2loss(gamma, threshold):  # threshold = b
    diff = (gamma - 0) ** 2
    weight = torch.where(diff > threshold ** 2, torch.ones_like(gamma), torch.zeros_like(gamma))
    diff_weighted = diff * weight
    return diff_weighted.mean()
def diff_range_loss(a, b, std):
    diff = (a - b) ** 2
    weight = torch.where(diff > std ** 2, torch.ones_like(a), torch.zeros_like(a))
    diff_weighted = diff * weight
    return diff_weighted.mean()

### main loss functions : 
# In all of these loss functions, only discriminator have been used as input
# The generator are not used here, we assume that the data_fake have been created else where and just use it here

def get_adversarial_loss(discriminator,data_real,data_fake,criterion,device=torch.device("cuda")):
    """
    : ####### Input #######
    :discriminator : Pos2DDiscriminator -> N x 5 
    :data_real : N x 16 x 2
    :data_fake : N x 16 x 2
    :criterion : nn.MSELoss(reduction='mean').to(device)
    : ####### Output #######
    :adversarial_loss : a scalar (combine the real_loss and fake_loss, usually used for discriminator training, but here they also use this for generator training)
    :generator_loss : a scalar (only use the fake_loss, usually used for training generator, but here they dont use it)

    """
    discriminator.to(device)
    data_real.to(device)
    data_fake.to(device)
    real_validity = discriminator(data_real) # shape N x 1
    fake_validity = discriminator(data_fake) # shape N x 1
    real_label = Variable(torch.ones(real_validity.size())).to(device) # shape N x 1
    fake_label = Variable(torch.zeros(fake_validity.size())).to(device) # shape N x 1
    # print(fake_label)
    adversarial_real_loss = criterion(real_validity,real_label)
    adversarial_fake_loss = criterion(fake_validity,fake_label)
    adversarial_loss = (adversarial_real_loss + adversarial_fake_loss)*0.5
    generator_loss = adversarial_fake_loss
    return adversarial_loss,generator_loss
def get_diff_loss(args,generator_output_dict):
    """
    control the modification range for:
    :input : generator_output_dict have 'pose_ba','ba_diff','pose_bl','blr','pose_rt','rt'
    :input : here we only use 'ba_diff', 'blr'
    :output : a loss - a scalar
    """
    diff_loss_dict = {} # will have two element : 'loss_diff_angle', 'loss_diff_blr'

    # regulation loss for bart to avoid gan collapse
    angle_diff = generator_output_dict['ba_diff']  # 'ba_diff' bx15;
    angle_diff_loss = diff_range_loss(torch.mean(angle_diff, dim=-1), args.ba_range_m, args.ba_range_w)

    diff_loss_dict['loss_diff_angle'] = angle_diff_loss.mean()

    blr = generator_output_dict['blr']

    blr_loss = rectifiedL2loss(blr, args.blr_limit)  # blr_limit

    diff_loss_dict['loss_diff_blr'] = blr_loss.mean()

    loss = 0
    for key in diff_loss_dict:
        loss = loss + diff_loss_dict[key]
    return loss
def get_classification_loss(x,classification_model,labels,criterion=nn.CrossEntropyLoss(),device=torch.device("cuda")): 
    """
    : input 0: x        : N,16,2
    : input 1: classification_model
    : input 2: labels   : N, 1
    : input 3 : criterion is likely nn.CrossEntropyLoss()
    : output : loss : a scalar
    """
    x.to(device)
    classification_model.to(device)
    labels.to(device)
    predicts = classification_model(x)
    classification_loss = criterion(predicts,labels)
    return classification_loss
def get_feedback_loss(args,classification_model,data_real,generator_output_dict,labels,current_epoch,device=torch.device("cuda"),criterion=nn.CrossEntropyLoss()):
    """
    INPUT:
    : args : hardratio_ba_s, hardratio_ba, 
    : args : hardratio_rt_s, hardratio_rt,
    : args : hardratio_std_ba, gloss_factordiv_ba, gloss_factorfeedback_ba
    : args : hardratio_std_rt, gloss_factordiv_rt, gloss_factorfeedback_rt
    : classification_model : is our classification model, should be set to train mode first
    : data_real : N x 16 x 2
    : generator_output_dict : we are going to use 'pose_ba' and 'pose_rt", no need to use 'pose_bl'
    : generator_output_dict['pose_ba'] : N x 16 x 2
    : generator_output_dict['pose_rt'] : N x 16 x 2 -> you can called it data_fake, since RT is the final augmenting ops
    : labels : N x 1
    """
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    # def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    # def fix_hardratio(target_std, target_mean, harder, easier, gloss_factordiv, gloss_factorfeedback, tag=''):
    #     harder_value = harder / easier # this is the one that is causing nan pain in the ass
    #     hard_std = torch.std(harder_value) # this will be nan, cause harder value is just a scalar number
    #     hard_mean = torch.mean(harder_value) # this will be nan, cause harder value is just a scalar number
    #     hard_div_loss = torch.mean((hard_std - target_std) ** 2)
    #     hard_mean_loss = diff_range_loss(harder_value, target_mean, target_std)
    #     result = hard_mean_loss * gloss_factorfeedback
    #     return result
    def fix_hardratio(target_std, hard_ratio, fake_loss, real_loss, gloss_factordiv, gloss_factorfeedback, tag=''):
        fake_loss = fake_loss.detach()
        real_loss = real_loss.detach()
        
        hard_loss = torch.abs(1.0 - torch.exp(fake_loss- hard_ratio * real_loss))
        return hard_loss


    # real_2d_pose -> classification model -> output 9 class
    real_loss = get_classification_loss(data_real,classification_model,labels,criterion,device)
    # fake_2d_pose_after_BA - > classification model -> output 9 class
    fake_loss_ba = get_classification_loss(generator_output_dict['pose_ba'],classification_model,labels,criterion,device)
    # fake_2d_pose_after_RT - > classification model -> output 9 class
    #fake_loss_rt = get_classification_loss(generator_output_dict['pose_rt'],classification_model,labels,criterion,device)
    fake_loss_bl = get_classification_loss(generator_output_dict['pose_bl'],classification_model,labels,criterion,device)
    # update the hard ratio for ba and rt , according to the current epoch
    hardratio_ba = update_hardratio(args.hardratio_ba_s, args.hardratio_ba, current_epoch, args.epochs)
    #hardratio_rt = update_hardratio(args.hardratio_rt_s, args.hardratio_rt, current_epoch, args.epochs)
    hardratio_bl = update_hardratio(args.hardratio_bl_s, args.hardratio_bl, current_epoch, args.epochs)
    # get feedback loss
    pos_fake_loss_baToReal = fix_hardratio(args.hardratio_std_ba, hardratio_ba,
                                             fake_loss_ba, real_loss,
                                             args.gloss_factordiv_ba, args.gloss_factorfeedback_ba, tag='ba')
    # pos_fake_loss_rtToReal  = fix_hardratio(args.hardratio_std_rt, hardratio_rt,
    #                                          fake_loss_rt, real_loss,
    #                                          args.gloss_factordiv_rt, args.gloss_factorfeedback_rt, tag='rt')
    pos_fake_loss_blToReal  = fix_hardratio(args.hardratio_std_bl, hardratio_bl,
                                             fake_loss_bl, real_loss,
                                             args.gloss_factordiv_bl, args.gloss_factorfeedback_bl, tag='bl')
    # feedback_loss = pos_fake_loss_baToReal + pos_fake_loss_rtToReal
    feedback_loss = (pos_fake_loss_baToReal + pos_fake_loss_blToReal)*0.5
    return feedback_loss
