import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    def __init__(self, mode, nms_iou=0.7, n_train_pre_nms=12000, n_train_post_nms=600, n_test_pre_nms=3000, n_test_post_nms=300, min_size=16):
        # 预测/训练模式
        self.mode = mode
        self.nms_iou = nms_iou

        # 训练用到的建议框数量
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms

        # 预测用到的建议框数量
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        # 建议框最小值
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor).type_as(loc)

        # 将RPN网络预测的偏移量结果转化成bbox
        roi = loc2bbox(anchor, loc)

        # 防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # 保留roi框宽高的最小值不小于16
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]

        # 将对应的建议框保留下来
        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序，取出建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        '''
        非极大抑制(nms)：降低目标检测算法输出中冗余边界框
            1.得分排序： 首先，对所有的边界框按照其检测得分（confidence score）进行降序排序，即从高到低排列。
            2.选择最高得分框： 选择得分最高的边界框，将其保留，并从候选框列表中移除。
            3.去除重叠框： 遍历剩余的边界框，去除与已选择的框有较大重叠（IoU大于设定阈值）的框。这样可以确保在同一个目标位置不会有多个边界框。
            4.重复步骤2和步骤3： 重复进行上述步骤，直到处理完所有的边界框。
        '''
        # 对建议框进行非极大抑制
        keep = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16, mode="training"):
        super(RegionProposalNetwork, self).__init__()

        # 高宽比ratio是指在 base_size(16) * anchor_scales[j]为基础下，h/w的值

        # 生成基础的先验框，shape为[9, 4]
        # shape[0] = 9 是因为生成了 len(ratios) * len(anchor_scales) = 3 * 3 个检测框
        # shape[1] = 4 是由于将矩形检测框描述为[上边, 左边, 下边, 右边]
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)

        # 获取基础检测框的数量，应该为9
        n_anchor = self.anchor_base.shape[0]

        # 用一个 3*3 的卷积整合图像的特征
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # 预测先验框内部是否包含物体
        # n_anchor * 2 表示每个锚框有两个得分（例如，表示目标存在和不存在的概率）
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # 回归预测对先验框进行调整
        # n_anchor * 4 表示每个锚框有四个坐标偏移（例如，表示目标的边界框的坐标）
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # 特征点间距步长
        # 控制在检测任务中使用的建议框的尺度，以及与输入图像的原始尺度之间的关系
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(mode)

        # FPN权值初始化（均值、方差）
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape

        x = F.relu(self.conv1(x))

        # 回归预测对先验框进行调整
        rpn_locs = self.loc(x)
        # 第一个维度表示批量大小，第二个维度表示特征点的总数（由原先的高度和宽度决定），第三个维度为 4，表示每个特征点对应的坐标偏移
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(x)
        # 第一个维度表示批量大小，第二个维度表示特征点的总数（由原先的高度和宽度决定），第三个维度为 2，表示每个特征点对应的两个得分（通常表示目标存在和不存在的概率）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        # 进行softmax概率计算，每个先验框只有两个判别结果
        # 内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        # 生成先验框，此时获得的anchor是布满特征网格点的，即每个特征点都会画出9个边界框
        # 当输入图片为600,600,3的时候，shape为(12996, 4): 600/16 * 600/16 * 9
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
