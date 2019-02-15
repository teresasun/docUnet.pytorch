from models.deeplab_models.backbone import xception
from models.deeplab_models.backbone import drn, mobilenet, resnet


def build_backbone(backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
