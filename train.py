import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn, get_lr)

from tqdm import tqdm


if __name__ == "__main__":
    pretrained = False

    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/voc_weights_vgg.pth'
    save_dir = 'logs/'
    train_annotation_path = 'VOCdevkit/2007_train.txt'
    val_annotation_path = 'VOCdevkit/2007_val.txt'

    class_names, num_classes = get_classes(classes_path)

    input_shape = [600, 600]

    '''
      当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
      [90, 180] ; [180, 360]; [360, 720]; [128, 128]; 
      [256, 256]; [512, 512]; [180, 90] ; [360, 180]; 
      [720, 360]
    '''
    anchors_size = [8, 16, 32]
    Epochs = 100
    batch_size = 4
    # 保存间隔时间
    save_period = 5

    # 模型的最大学习率
    Init_lr = 1e-4
    # 模型的最小学习率
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0

    # 在训练时进行评估
    eval_flag = True
    # 评估间隔时间
    eval_period = 5

    Cuda = True
    train_gpu = [0, ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    seed = 11
    seed_everything(seed)

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # 记录Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    print(model)
    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape,
        Epochs=Epochs, batch_size=batch_size,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        save_period=save_period, save_dir=save_dir, num_train=num_train, num_val=num_val
    )

    # 冻结bn层
    model.freeze_bn()

    # 自适应调整学习率
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    # 学习率下降的公式（cos）
    lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, Epochs)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True,
                     collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True,
                         collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    train_util = FasterRCNNTrainer(model_train, optimizer)

    # 绘制map曲线
    eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                 eval_flag=eval_flag, period=eval_period)

    # 开始模型训练
    for epoch in range(0, Epochs):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        total_loss = 0
        rpn_loc_loss = 0
        rpn_cls_loss = 0
        roi_loc_loss = 0
        roi_cls_loss = 0

        val_loss = 0
        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                images, boxes, labels = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    if Cuda:
                        images = images.cuda()

                rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1)

                total_loss += total.item()
                rpn_loc_loss += rpn_loc.item()
                rpn_cls_loss += rpn_cls.item()
                roi_loc_loss += roi_loc.item()
                roi_cls_loss += roi_cls.item()

                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'rpn_loc': rpn_loc_loss / (iteration + 1),
                                    'rpn_cls': rpn_cls_loss / (iteration + 1),
                                    'roi_loc': roi_loc_loss / (iteration + 1),
                                    'roi_cls': roi_cls_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

        print('Finish Train')
        print('Start Validation')
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                images, boxes, labels = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    if Cuda:
                        images = images.cuda()

                    train_util.optimizer.zero_grad()
                    _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)
                    val_loss += val_total.item()

                    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                    pbar.update(1)

        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # 保存权值
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    loss_history.writer.close()
