import torch
from torch import nn
import torch.nn.functional as F
from math import ceil


def _RoundChannels(c, divisor=8, min_value=None):
    """
    Округляет количество каналов в MBConv блоках
    """
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        """Squeeze and excite модуль, предназначенный для выборочного усиления
        и подавления определенных каналов

        Parameters
        ----------
        channels : int
            Количество входных и выходных каналов
        squeeze_channels : int
            Количество каналов для сжатия
        se_ratio : float
            Множитель для расчета количества промежуточных каналов
            squeeze_channels = squeeze_channels * se_ratio
        """
        super().__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = nn.SiLU()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()


    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y
    

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        """MBConvBlock - основной блок для построения модели EfficientNet. Состоит из фазы расширения количества каналов, свертки,
        модуля SqueezeAndExcite и фазы сужения количества каналов. При stride = 1 сопровождается разрывом связи и суммированием
        входного тензора с выходным тензором.

        Parameters
        ----------
        in_channels : int
            Количество входных каналов
        out_channels : int
            Количество выходных каналов
        kernel_size : int
            Размер ядра DepthwiseConv2D
        stride : int
            Шаг свертки
        expand_ratio : int
            Коэффициент для расчета количества каналов для расширения
        se_ratio : float
            Множитель для расчета количества промежуточных каналов в 
            модуле SqueezeAndExcite
        drop_path_rate : float
            Вероятность dropout перед суммированием в residual соединении
        """
        super().__init__()
        expand = (expand_ratio != 1)  # whether to expand input
        expand_channels = in_channels * expand_ratio  # compute expanded channels
        se = (se_ratio != 0.0)  # whether to use squeeze and excitation block
        self.residual_connection = (stride == 1 and in_channels == out_channels)  # whether to use residual connection
        self.dropout = nn.Dropout2d(drop_path_rate)

        conv = []  # forward pass

        if expand:  # expansion convolution projection
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.SiLU()
            )
            conv.append(pw_expansion)

        # main depthwise convolution
        dw = nn.Sequential(
            nn.Conv2d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=expand_channels,
                bias=False
            ),
            nn.BatchNorm2d(expand_channels),
            nn.SiLU()
        )
        conv.append(dw)

        # squeeze and excitation block
        if se:
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection to output channels
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)  # build forward pass


    def forward(self, x):
        # using residual connection if needed
        if self.residual_connection:
            return x + self.dropout(self.conv(x))
        else:
            return self.conv(x)
        

class EfficientNet(nn.Module):
    # base config to further rescale
    config = [
        # (in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]
    # dictionary with model type config
    net_param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }


    def __init__(self, model_type: str, stem_channels=32, drop_connect_rate=0.2):
        """Семейство моделей EfficientNetB0 - B7

        Parameters
        ----------
        model_type : string
            Тип модели для построения, доступные варианты:
            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
            'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
            'efficientnet-b6', 'efficientnet-b7'
        stem_channels : int, optional
            Количество каналов в первом сверточном слое, by default 32
        drop_connect_rate : float, optional
            Вероятность dropout перед суммированием в residual соединении, 
            by default 0.2
        """
        super().__init__()
        param = self.net_param[model_type]
        # scaling width from base config
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = _RoundChannels(stem_channels * width_coefficient)
            for conf in self.config:
                conf[0] = _RoundChannels(conf[0] * width_coefficient)
                conf[1] = _RoundChannels(conf[1] * width_coefficient)
        
        # scaling depth from base config
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = ceil(conf[6] * depth_coefficient)

        # build stem convolution
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU()
        )

        # count total number of blocks
        total_blocks = 0
        for conf in self.config:
            total_blocks += conf[6]

        # build core MBConv blocks
        self.blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in self.config:
            block = []
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(block) / total_blocks)
            block.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats - 1):
                drop_rate = drop_connect_rate * (len(block) / total_blocks)
                block.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
            self.blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(self.blocks)


    def forward(self, x):
        out = []  # out tensors
        x = self.stem_conv(x)  # pass through stem
        # get outputs with strides 8, 16, 32
        out_blocks = [1, 2, 4, 6]  # number of blocks needed
        for i, block in enumerate(self.blocks):
            x = block(x)  # forward pass through block
            if i in out_blocks:
                out.append(x)

        return out


class DecodeBlock(nn.Module):
    def __init__(self, config: list):
        """Decode block of Dense UNet

        Parameters
        ----------
        config : list
            List with each decoder block config [stride, deconv, deconv_ch, conv_ch],
            where stride -> stride of feature map from backbone : int
            deconv -> whether to use Transpose Conv for upscaling : bool
            deconv_ch -> number of channels in Transpose Conv (if deconv=True) : int
            conv_ch -> number of channels in Conv layers
        """
        super().__init__()
        stride, deconv, deconv_ch, conv_ch = config
        if deconv:
            self.upscale = nn.Sequential(
                nn.ConvTranspose2d(deconv_ch, conv_ch, 2, 2),
                nn.BatchNorm2d(conv_ch),
                nn.SiLU()
            )
            in_ch = conv_ch * 2
        else:
            self.upscale = nn.Upsample(scale_factor=2)
            in_ch = conv_ch + deconv_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, conv_ch, 3, 1, 1),
            nn.BatchNorm2d(conv_ch),
            nn.SiLU()
        )


    def forward(self, x, y):
        x = self.upscale(x)  # perform upsampling
        x = torch.cat((x, y), dim=1)  # concatenate with next fmap
        x = self.conv(x)  # perform conv
        return x

        
        
        


class UNetHead(nn.Module):
    def __init__(self, config=[[32, False, 320, 112],
                               [16, False, 112, 40],
                               [8, False, 40, 24]]):
        """Decode head for Dense UNet

        Parameters
        ----------
        config : list
            List with each decoder block config [stride, deconv, deconv_ch, conv_ch],
            where stride -> stride of feature map from backbone : int
            deconv -> whether to use Transpose Conv for upscaling : bool
            deconv_ch -> number of channels in Transpose Conv (if deconv=True) : int
            conv_ch -> number of channels in Conv layers
        """
        super().__init__()
        self.decoders = []
        self.upscale = nn.Upsample(scale_factor=4)
        for conf in config:  # get decoders
            self.decoders.append(DecodeBlock(conf))
        self.decoders = nn.ModuleList(self.decoders)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(config[-1][-1], 1, 1)
        

    def forward(self, x):
        # x - feature maps from backbone
        # fuse 2 last feature maps with first decoder
        y = self.decoders[0](x[-1], x[-2])
        for i in range(1, len(self.decoders)):  # fuse all features
            y = self.decoders[i](y, x[-2 - i])
        y = self.upscale(y)
        y = self.conv(y)
        y = self.sigmoid(y)
        return y
    

class EfficientUnet(nn.Module):
    def __init__(self, model_type='efficientnet-b0'):
        """Segmentation model with EfficientNet backbone and 
        UNet decoder

        Parameters
        ----------
        model_type : str, optional
            Model type for backbone, by default 'efficientnet-b0'
        """
        super().__init__()
        self.backbone = EfficientNet(model_type)
        self.head = UNetHead()


    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x