import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1Loss(nn.Module):
    def __init__(self, weight=1):
        super(L1Loss, self).__init__()
        self.weight = weight
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        return self.weight * self.loss(pred, gt)


class LGCLoss(nn.Module):
    def __init__(self, lgc_weight=0.1):
        super(LGCLoss, self).__init__()
        self.lambda_lgc = lgc_weight

    def forward(self, feat_masked, feat_inpaint, feat_gt):
        lgc_loss = 0.0
        for f_m, f_i, f_g in zip(feat_masked, feat_inpaint, feat_gt):
            l_mi = torch.mean(torch.abs(f_m - f_i))
            l_mg = torch.mean(torch.abs(f_m - f_g))
            l_ig = torch.mean(torch.abs(f_i - f_g))

            l_perc = (l_mi + l_mg + l_ig)/3

            lgc_loss += l_perc

        lgc_loss /= len(feat_gt)

        return self.lambda_lgc * lgc_loss
