import torch
import torch.nn as nn
import torch.nn.functional as F

class Poly_cross_entropy(nn.Module):
    def __init__(self,weight=None,reduction='mean'):
        super(Poly_cross_entropy,self).__init__()

        if weight is None:
            self.cel = nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.cel = nn.CrossEntropyLoss(reduction=reduction,weight=weight)

        self.softmax = nn.Softmax(1)
        
    def forward(self,logits, labels, epsilon=1.0):
        labels = labels.to(torch.float).unsqueeze(1)
        # logits = logits.argmax(dim=1).to(torch.float).unsqueeze(1)
        poly1 = torch.sum(labels * self.softmax(logits), dim=1)
        ce_loss = self.cel(logits, labels)
        pce_loss = ce_loss + epsilon * (1 - poly1)

        return torch.mean(pce_loss)

class Poly_BCE(nn.Module):
    def __init__(self,weight=None,logits=False,reduction='mean'):
        super(Poly_BCE,self).__init__()

        self.cel = nn.BCELoss(reduction=reduction,weight=weight)
        self.logits = logits
        self.reduction = reduction
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,logits, labels, epsilon=1.0):
        labels = labels.to(torch.float)
        poly1 = torch.sum(labels * self.softmax(logits), dim=-1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(logits, labels, reduction=self.reduction)
        pce_loss = BCE_loss + epsilon * (1 - poly1)

        return torch.mean(pce_loss)
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.CELoss = torch.nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        else:
            # inputs = self.softmax(inputs)
            # BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
            BCE_loss = self.CELoss(inputs,targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
       
        return F_loss
    
class FocalDiceLoss(nn.Module):
    """Multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight
        self.softmax = nn.Softmax(1)

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        # 多分类/多标签
        # predict = nn.Sigmoid()(input)
        # 二分类
        predict = self.softmax(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))