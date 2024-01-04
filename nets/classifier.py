import warnings

import torch
from torch import nn

warnings.filterwarnings("ignore")


class Classify(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale):
        super(Classify, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )

        # 对ROIPooling后的的结果进行回归预测
        # 每个类别都对应四个值，用于表示边界框的位置偏移（左上角 x、左上角 y、右下角 x、右下角 y）
        self.cls_loc = nn.Linear(4096, n_class * 4)

        # 对ROIPooling后的的结果进行分类
        self.score = nn.Linear(4096, n_class)

        # 权值初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, x, pool):
        n = x.shape[0]

        # 利用classifier网络进行特征提取
        pool = pool.view(pool.size(0), -1)

        # 池化了一次，故图片大小 600 被变为 300
        # 当输入为一张图片的时候，这里获得的fc7的shape为[300, 4096]
        fc7 = self.classifier(pool)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
