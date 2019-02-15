# -*- coding: utf-8 -*-
# @Time    : 18-11-27 下午1:11
# @Author  : zhoujun

trainroot='/data2/zj/data/add_bg_img_800_600'
output_dir = 'output/deeplab_add_bg_img_800_600_item_origin_deeplab_drn'

gpu_id = 2
workers = 6
start_epoch = 0
epochs = 100

train_batch_size = 3
back_step = 10

lr = 1e-4
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 50
display_interval = 100
restart_training = True
checkpoint = 'output/deeplab_add_bg_img_800_600_item_origin_deeplab_resnet_deconv/DocUnet_26_2.20041725584507.pth'

# random seed
seed = 2