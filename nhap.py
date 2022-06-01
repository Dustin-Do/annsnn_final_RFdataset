import torch.nn as nn

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


# Define model VGG
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):  # VGG('VGG16', num_classes, dropout)
        super(VGG, self).__init__()
        self.init_channels = 3
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)  # [64, 64, 'M']
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)  # [128, 128, 'M']
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)  # [256, 256, 256, 'M']
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)  # [512, 512, 512, 'M']
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)  # [512, 512, 512, 'M']
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),  # Multiplicate input with matrix weight has size of [512, 4096]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)  # 'num_classes': number of classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Fills the input Tensor with values according to the method described in Delving deep into rectifiers:
                # Surpassing human-level performance on ImageNet classification
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)  # Fills 'm.weight' with the value 'val'
                nn.init.zeros_(m.bias)  # Fills 'm.bias' with 0
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                # 'in_channels' (int) – Number of channels in the input image.
                # 'out_channels' (int) – Number of channels produced by the convolution.
                # 'stride' (int or tuple, optional) – Stride of the convolution.
                # 'padding' (int, tuple or str, optional) – Padding added to all four sides of the input.
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))

                # Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel
                # dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by
                # reducing internal covariate shift .
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_normed, self).__init__()
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)

    def _make_layers(self, cfg, dropout):
        layers = []
        for i in range(5):
            for x in cfg[i]:
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(3, x, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    self.init_channels = x
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)


def vgg11(num_classes=10, dropout=0, **kargs):
    return VGG('VGG11', num_classes, dropout)


def vgg13(num_classes=10, dropout=0, **kargs):
    return VGG('VGG13', num_classes, dropout)


def vgg16(num_classes=10, dropout=0, **kargs):
    return VGG('VGG16', num_classes, dropout)


def vgg19(num_classes=10, dropout=0, **kargs):
    return VGG('VGG19', num_classes, dropout)


def vgg16_normed(num_classes=10, dropout=0, **kargs):
    return VGG_normed('VGG16', num_classes, dropout)