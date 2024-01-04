import numpy as np


# 生成基础的先验框
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    # 生成4维的矩阵是由于找到的锚点是矩形框的中心点，分别为[上边, 左边, 下边, 右边]
    # 高宽比ratio是指在 base_size * anchor_scales[j]为基础下，h/w的值
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


# 将基础先验框对应到所有特征点上
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算特征点网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 生成形状为 (K, A, 4) 的张量 anchor。其中，第一个维度表示特征点的数量，第二个维度表示锚框的数量，第三个维度表示每个锚框的坐标
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride = 38, 38, 16
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)
    box_widths = anchors_all[:, 2] - anchors_all[:, 0]
    box_heights = anchors_all[:, 3] - anchors_all[:, 1]

    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0], anchors_all[i, 1]], box_widths[i], box_heights[i], color="r",
                             fill=False)
        ax.add_patch(rect)
    plt.show()
