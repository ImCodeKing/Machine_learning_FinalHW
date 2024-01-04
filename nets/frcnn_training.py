import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


'''
根据 Anchor 和 Ground Truth Boxes 之间的 IoU关系，为 RPN 网络生成训练目标。
在目标生成过程中，正样本是与 Ground Truth 有较高 IoU 的 Anchor，负样本是与 Ground Truth 有较低 IoU 的 Anchor。
通过调整构造函数中的参数，可以灵活控制正负样本的生成策略。
'''
class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        # 正样本阈值
        self.pos_iou_thresh = pos_iou_thresh
        # 负样本阈值
        self.neg_iou_thresh = neg_iou_thresh
        # 正样本在总样本数占比
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            # 先验框向真实框调整的建议值
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        # anchor和bbox的iou（两个边界框相交面积与它们的并集面积之比）
        ious = bbox_iou(anchor, bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))

        # 获得每一个先验框最对应的真实框（在bbox求出来的iou里找最大）
        argmax_ious = ious.argmax(axis=1)

        # 先验框对应的真实框的iou
        max_ious = np.max(ious, axis=1)

        # 真实框对应的先验框（在anchor求出来的iou里找最大）
        gt_argmax_ious = ious.argmax(axis=0)

        # 保证每一个真实框都存在对应的先验框
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个先验框对应的最大的真实框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # 正样本是包含物体
        # 真实框对应的先验框一定包含物体
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        # 限制正样本数量小于等于128，超出的置为-1（忽略）
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 平衡正负样本
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        # 每张图像中作为正样本的 RoI 的数量每张图像中作为正样本的 RoI 的数量
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)

        # 计算建议框和真实框的重合程度
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # 获得每一个建议框最对应的真实框
            gt_assignment = iou.argmax(axis=1)
            # 获得每一个建议框最对应的真实框的iou
            max_iou = iou.max(axis=1)
            # 真实框的标签要+1因为有背景的存在
            gt_roi_label = label[gt_assignment] + 1

        # 建议框和真实框重合程度大于阈值的作为正样本
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # 建议框和真实框重合程度在负阈值上下界作为负样本
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # 选取平衡的正负样例
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train = model_train
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        # 为每个 anchor 分配标签
        self.anchor_target_creator = AnchorTargetCreator()
        # 为每个 RoI 分配分类标签和回归目标
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    '''
    计算预测边界框与真实边界框的位置差异 regression_diff。
    将位置差异取绝对值，并将其转换为浮点数。
    使用 torch.where 函数，当位置差异小于 1/sigma^2 时，采用 Huber Loss 公式的平方项；当位置差异大于等于 1/sigma^2 时，采用 Huber Loss 公式的线性项。
    最后，将损失在所有正样本上进行求和，并除以正样本的数量，得到平均损失
    '''
    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def forward(self, imgs, bboxes, labels, scale):
        # n = batch_size
        n = imgs.shape[0]
        img_size = imgs.shape[2:]

        # vgg16 提取图像特征（Conv layers）
        base_feature = self.model_train(imgs, mode='extractor')

        # 利用rpn网络获得网格特征点调整参数、得分（目标存在和不存在的概率）、rpn标注框、先验框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x=[base_feature, img_size], scale=scale, mode='rpn')

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []
        for i in range(n):
            # 将batch里的元素提取成单独的
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[i]

            # gt_rpn_loc 是利用真实框和先验框获得的网络应该有的建议框（先验框的偏移量）
            # gt_rpn_label 是先验框的标签（有无物品）
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()

            # rpn_loc_loss 是建议框网络选取锚框的损失
            # rpn_cls_loss 是网络在建议框标签分类的损失（这个标签是有无物品的标签）
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss

            # sample_roi 是所有建议框
            # gt_roi_loc 是网络产生的建议框（建议框的偏移量）
            # gt_roi_label 是建议框的标签（有无物品）
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

        sample_rois = torch.stack(sample_rois, dim=0)
        sample_indexes = torch.stack(sample_indexes, dim=0)

        # Roi pooling
        pool = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode='RoIPooling')

        # roi_cls_locs 是边界框的偏移
        roi_cls_locs, roi_scores = self.model_train([base_feature, pool], mode='classify')

        for i in range(n):
            # Roi pooling 之后，取出对应图片的回所有建议框
            n_sample = roi_cls_locs.size()[1]

            roi_cls_loc = roi_cls_locs[i]
            roi_score = roi_scores[i]
            gt_roi_loc = gt_roi_locs[i]
            gt_roi_label = gt_roi_labels[i]

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # roi_loc_loss 是Classifier网络更精确检测框的损失
            # roi_cls_loss 是Classifier网络给标签分类的损失（此处的标签是“物品名称”）
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()

        losses = self.forward(imgs, bboxes, labels, scale)
        losses[-1].backward()
        self.optimizer.step()

        return losses


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
