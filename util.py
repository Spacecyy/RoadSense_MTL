import os
import torch
from torch.autograd import grad
import torch.nn as nn


def compute_wasserstein(features, btch_sz, discriminator, use_gp = True, gp_weight = 0.1):

    num_domains = int(len(features)/btch_sz)
    dis_loss = 0
    for t in range(num_domains):

        for k in range(t + 1, num_domains):
            
            features_t = features[t * btch_sz:(t + 1) * btch_sz]
            features_k = features[k * btch_sz:(k + 1) * btch_sz]
            
            # print('!!!!!!!!!!!!!!!!features_t size', features_t.size())
            dis_t = discriminator(features_t)
            dis_k = discriminator(features_k)
            
            if use_gp:
                gp = gradient_penalty(discriminator, features_t, features_k)
                disc_loss = dis_t.mean() - dis_k.mean() - gp_weight*gp
            else: 
                disc_loss = dis_t.mean() - dis_k.mean()



            dis_loss += disc_loss
    
    return dis_loss

def compute_H_divergence_replace(features, btch_sz, discriminator, device):

    num_domains = int(len(features)/btch_sz)
    dis_loss = 0
    loss_fn = nn.BCELoss()

    for t in range(num_domains):
        features_t = features[t * btch_sz:(t + 1) * btch_sz]
        dis_t = discriminator(features_t)
        domain_label_t = torch.ones(len(features_t), 1).float().to(device)
        loss_t = loss_fn(dis_t, domain_label_t)
            
        for k in range(num_domains):
            if k == t:
                continue

            features_k = features[k * btch_sz:(k + 1) * btch_sz]
            dis_k = discriminator(features_k)
            domain_label_k = torch.zeros(len(features_k), 1).float().to(device)        
            loss_k = loss_fn(dis_k, domain_label_k)
            dis_loss += 0.5 * (loss_t + loss_k)      

    return dis_loss

def compute_H_divergence(features_list, task_idx, indice, discriminator, device):

    dis_loss = 0
    loss_fn = nn.BCELoss()


    fea_t = features_list[task_idx].view(features_list[task_idx].size(0), -1)
    # print('fea.shape: ', fea_t.shape)
    dis_t = discriminator(fea_t)
    domain_label_t = torch.ones(len(fea_t), 1).float().to(device)
    loss_t = loss_fn(dis_t, domain_label_t)

    for k in indice:
        if k == task_idx:
            continue
        fea_k = features_list[k].view(features_list[k].size(0), -1)
        dis_k = discriminator(fea_k)
        domain_label_k = torch.zeros(len(fea_k), 1).float().to(device)        
        loss_k = loss_fn(dis_k, domain_label_k)

        dis_loss += 0.5 * (loss_t + loss_k)

    return dis_loss

def compute_orthogonality(feature_share, feature_private):
    feature_share = feature_share.squeeze()
    feature_private = feature_private.squeeze()

    diff_loss = torch.matmul(feature_share.t(), feature_private)
    # print(feature_share.shape, feature_private.shape, diff_loss.shape)
    diff_loss = torch.sum(torch.square(diff_loss))
    # print(diff_loss.shape)
    return diff_loss

# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s

    interpolates = h_s + (alpha * differences)

    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


def sort_loaders(loaders_list, reverse=True):
    num_loaders = len(loaders_list)
    
    tuple_ = []
    
    for i in range(num_loaders):
        tuple_.append((loaders_list[i],len(loaders_list[i])))
    
    return sorted(tuple_, key=lambda tuple_len: tuple_len[1],reverse=reverse)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    # print(perm)
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        # print('in minibatches, xi.shape =', xi.shape)
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        
        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
    # print('pairs[0] ', pairs[0])

    return pairs

def random_mix_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    # print(perm)
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        if i < (len(minibatches) - 2):
            k = i + 2
        elif i == len(minibatches) - 2:
            k = 0
        else:
            k = 1

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        xk, yk = minibatches[perm[k]][0], minibatches[perm[k]][1]

        x_sample = torch.cat((xi, xj, xk), 0)
        y_sample = torch.cat((yi, yj, yk), 0)
        idx = torch.randperm(x_sample.size(0))
        
        x_sample = x_sample[idx]
        y_sample = y_sample[idx]
        

        x_sample1 = torch.narrow(x_sample, 0, 0, xi.size(0))
        y_sample1 = torch.narrow(y_sample, 0, 0, yi.size(0))
        

        x_sample2 = torch.narrow(x_sample, 0, xi.size(0), xi.size(0))
        y_sample2 = torch.narrow(y_sample, 0, yi.size(0), yi.size(0))
        
        pairs.append(((x_sample1, y_sample1), (x_sample2, y_sample2)))
    
    return pairs

def random_pairs_of_minibatches_(all_ｘ, all_y):
    perm = torch.randperm(len(all_x)).tolist()
    pairs = []

    for i in range(len(all_x)):
        j = i + 1 if i < (len(all_x) - 1) else 0

        xi, yi = all_ｘ[perm[i]], all_y[perm[i]]
        xj, yj = all_ｘ[perm[j]], all_y[perm[j]]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg['scale_pos']
        self.scale_neg = cfg['scale_neg']

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class SelfNegSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(SelfNegSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg['scale_pos']
        self.scale_neg = cfg['scale_neg']

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            
            # modified!!!
            loss.append(neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss
    

class SelfPosSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(SelfPosSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg['scale_pos']
        self.scale_neg = cfg['scale_neg']

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            
            # modified!!!
            loss.append(pos_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss
    

def get_num_batches(num_domains, minibatches):

    num = []
    for i in range(num_domains):
        num.append(len(minibatches[i]))
    return min(num)



def shuffle_batch(all_x, all_y):

    #print('!!!!all y', all_y)
    perm = torch.randperm(len(all_x)).tolist()
    pairs = []

    for i in range(len(all_x)):

        xi, yi = all_ｘ[perm[i]], all_y[perm[i]]
        pairs.append((xi, yi))


    return pairs


def shuffle_minibatch(all_x, all_y):
    perm = torch.randperm(len(all_x)).tolist()

    x_new, y_new = [], []

    for i in range(len(all_x)):
        x_new.append(all_x[perm[i]])
        y_new.append(all_y[perm[i]])
    
    return x_new, y_new


def shuffle_batch_label(all_x, all_y, all_z):
    
    perm = torch.randperm(len(all_x)).tolist()
    pairs = []

    for i in range(len(all_x)):

        xi, yi, zi = all_ｘ[perm[i]], all_y[perm[i]], all_z[perm[i]]
        pairs.append((xi, yi, zi))


    return pairs


def pair_wise_wasserstein(features_t, features_k, discriminator, use_gp = True, gp_weight = 0.1):

    dis_loss = 0

    dis_t = discriminator(features_t)
    dis_k = discriminator(features_k)
            
    if use_gp:
        gp = gradient_penalty(discriminator, features_t, features_k)
        disc_loss = dis_t.mean() - dis_k.mean() - gp_weight*gp
    else: 
        disc_loss = dis_t.mean() - dis_k.mean()

    dis_loss += disc_loss
    
    return dis_loss    
