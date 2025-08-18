import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):

        loss = multilabel_categorical_crossentropy(labels,logits)
        loss = loss.mean()
        return loss
    
class HingeABL(nn.Module):
    def __init__(self, m=5):
        super().__init__()
        self.m = m
        
    def forward(self, logits, labels):
        """
        HingeABL
        """
        m = self.m
        p_num = labels[:, 1:].sum(dim=1)

        p_logits_diff = logits[:, 0].unsqueeze(dim=1) - logits
        p_logits_imp = F.relu(p_logits_diff + m)
        p_logits_imp = p_logits_imp * labels
        p_logits_imp = p_logits_imp[:, 1:]
        p_logits_imp = p_logits_imp / (p_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

        n_logits_diff = logits - logits[:, 0].unsqueeze(dim=1)
        n_logits_imp = F.relu(n_logits_diff + m)
        n_logits_imp = n_logits_imp * (1 - labels)
        n_logits_imp = n_logits_imp[:, 1:]
        n_logits_imp = n_logits_imp / (n_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

        exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))   # margin=5

        p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
        n_prob = exp_th / (exp_th + torch.exp(logits))

        p_item = -torch.log(p_prob + 1e-30) * labels
        p_item = p_item[:, 1:] * p_logits_imp
        n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
        n_item = n_item[:, 1:] * n_logits_imp

        p_loss = p_item.sum(1)
        n_loss = n_item.sum(1)
        loss = p_loss + n_loss
        loss = loss.mean()
        return loss


class AFLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=1.0):
        super().__init__()
        threshod = nn.Threshold(0, 0)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg


    def forward(self, logits, labels):
        # Adapted from Focal loss https://arxiv.org/abs/1708.02002, multi-label focal loss https://arxiv.org/abs/2009.14119
        # TH label 
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        label_idx = labels.sum(dim=1)

        two_idx = torch.where(label_idx==2)[0]
        pos_idx = torch.where(label_idx>0)[0]

        neg_idx = torch.where(label_idx==0)[0]
     
        p_mask = labels + th_label
        n_mask = 1 - labels
        neg_target = 1- p_mask
        
        num_ex, num_class = labels.size()
        num_ent = int(np.sqrt(num_ex))
        # Rank each positive class to TH
        logit1 = logits - neg_target * 1e30
        logit0 = logits - (1 - labels) * 1e30

        # Rank each class to threshold class TH
        th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))

        # Probability of relation class to be positive (1)
        prob_1 = probs[:, 0 ,:]
        # Probability of relation class to be negative (0)
        prob_0 = probs[:, 1 ,:]
        prob_1_gamma = torch.pow(prob_1, self.gamma_neg)
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0 ,:]
        log_prob_0 = log_probs[:, 1 ,:]
        
        
        


        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        rank2 = F.log_softmax(logit2, dim=-1)

        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
        
        loss2 = -(rank2 * th_label).sum(1) 


        

        loss =  1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
        

        return loss
    
class SATLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels):
        """
        MeanSAT
        """
        exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))

        p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
        n_prob = exp_th / (exp_th + torch.exp(logits))

        p_num = labels[:, 1:].sum(dim=1)
        n_num = 96 - p_num

        p_item = -torch.log(p_prob + 1e-30) * labels
        p_item = p_item[:, 1:]
        n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
        n_item = n_item[:, 1:]

        p_loss = p_item.sum(1) / (p_num + 1e-30)
        n_loss = n_item.sum(1) / (n_num + 1e-30)
        loss = p_loss + n_loss
        loss = loss.mean()
        return loss


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss
