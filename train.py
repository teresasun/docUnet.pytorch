# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import torch.utils.data as Data
from torchvision import transforms
from MyDataSet import ImageData
from models.deeplab_models.deeplab import DeepLab  # 一个deeplab v3+网络
import time
import config
from tensorboardX import SummaryWriter
from loss import DocUnetLoss_DL_batch as DocUnetLoss
import os
import shutil

torch.backends.cudnn.benchmark = True

def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger

def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return start_epoch


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = ImageData(config.trainroot, transform=transforms.ToTensor(), t_transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=False,
                                   num_workers=int(config.workers))

    writer = SummaryWriter(config.output_dir)
    # net = Doc_UNet(n_channels=3, n_classes=2)
    net = DeepLab(backbone='resnet', output_stride=16, num_classes=2, pretrained=True)
    # net = drn_c_58(BatchNorm=torch.nn.BatchNorm2d, num_classes=2, pretrained=True)
    net = net.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(model=net, input_to_model=dummy_input)

    criterion = DocUnetLoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        # net.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage.cuda(0)))
        start_epoch = load_checkpoint(config.checkpoint, net, optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay,last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay)

    all_step = len(train_loader)
    epoch = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            net.train()
            train_loss = 0.
            start = time.time()
            scheduler.step()
            for i, (images, labels) in enumerate(train_loader):
                # if float(scheduler.get_lr()[0]) > opt.end_learning_rate:
                #     scheduler.step()

                images, labels = images.to(device), labels.to(device)
                # Forward
                y1 = net(images)
                loss = criterion(y1, labels)
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                cur_step = epoch * all_step + i
                writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=cur_step)
                writer.add_scalar(tag='Train/lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)

                if (i + 1) % config.display_interval == 0:
                    batch_time = time.time() - start
                    logger.info('[%d/%d], [%d/%d], batch_loss: %.4f, time:%0.4f, lr:%s' % (
                        epoch + 1, config.epochs, (i + 1), all_step, loss.item() / config.train_batch_size, batch_time,
                        str(scheduler.get_lr()[0])))
                    start = time.time()

            logger.info('[%d/%d], train_loss: %.4f, time:%0.4f, lr:%s' % (
                epoch + 1, config.epochs, train_loss / train_data.__len__(), time.time() - start,
                str(scheduler.get_lr()[0])))
            save_checkpoint(
                '{}/DocUnet_{}_{}.pth'.format(config.output_dir, epoch + 1, train_loss / train_data.__len__()),
                net, optimizer, epoch + 1)
        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), net, optimizer, epoch + 1)


if __name__ == '__main__':
    train()
