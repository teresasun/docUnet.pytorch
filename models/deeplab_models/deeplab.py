import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab_models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab_models.aspp import build_aspp
from models.deeplab_models.decoder import build_decoder
from models.deeplab_models.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, pretrained=True,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        assert backbone in ['resnet', 'xception', 'drn', 'mobilenet'], \
            "backbone must in [resnet', 'xception', 'drn', 'mobilenet']"
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, pretrained=pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, output_padding=1),
                                    nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, output_padding=1)
                                    )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # x = self.deconv(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=2, pretrained=False)
    model.eval()
    input = torch.rand(1, 3, 600, 800)
    output = model(input)
    print(output.size())
