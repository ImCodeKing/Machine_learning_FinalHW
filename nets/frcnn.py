import torch.nn as nn

from nets.classifier import Classify
from nets.rpn import RegionProposalNetwork
from nets.RoIPooling import RoIPoolingHead
from nets.vgg16 import sep_vgg16


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, mode="training", feat_stride=16, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2], pretrained=False):
        super(FasterRCNN, self).__init__()
        # 存储特征图的步长
        self.feat_stride = feat_stride

        # 使用vgg16作为主干网络（Conv layers）
        self.extractor = sep_vgg16(pretrained)

        # 构建建议框网络（Region Proposal Networks）
        self.rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            mode=mode
        )

        # Roi池化网络（Roi Pooling）
        self.RoI_pooling = RoIPoolingHead(
            roi_size=7,
            spatial_scale=1,
        )

        # 分类器网络（Classification）
        self.classify = Classify(
            n_class=num_classes + 1,
            roi_size=7,
            spatial_scale=1
        )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            # [height, width]
            img_size = x.shape[2:]

            # vgg16 提取图像特征（Conv layers）
            base_feature = self.extractor.forward(x)

            # 获得标注框（Region Proposal Networks）
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)

            # Roi池化并分类（Roi Pooling + Classification）
            pool = self.RoI_pooling.forward(base_feature, rois, roi_indices, img_size)
            roi_cls_locs, roi_scores = self.classify.forward(base_feature, pool)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            # vgg16 提取图像特征（Conv layers）
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # 获得标注框（Region Proposal Networks）
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "RoIPooling":
            base_feature, rois, roi_indices, img_size = x
            # Roi池化（Roi Pooling）
            pool = self.RoI_pooling.forward(base_feature, rois, roi_indices, img_size)
            return pool
        elif mode == "classify":
            base_feature, pool = x
            # 分类（Classification）
            roi_cls_locs, roi_scores = self.classify.forward(base_feature, pool)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
