import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, cardinality=1, 
            dilation=1, first_dilation=None, act_layer=nn.ReLU, aa_layer=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        first_planes = planes# // reduce_first
        outplanes = planes# * self.expansion
        padding = 2
        first_dilation = dilation#first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=5, stride=1 if use_aa else stride, padding=padding,
            dilation=first_dilation, bias=False)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=5, stride=1 if use_aa else stride, padding=padding,
            dilation=first_dilation, bias=False)

        self.act2 = act_layer()
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        
        x = x + shortcut

        return x


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32, drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 #if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=None, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            # downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx# / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class OffResNet2d(nn.Module):
    def __init__(
            self, block, layers, output_stride=32,
            cardinality=1, base_width=128, stem_width=128, stem_type='', block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., block_args=None, **kwargs):
        super(OffResNet2d, self).__init__()
        
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])
        
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width # stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width, stem_width)
            self.conv1 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer()])
            self.conv2 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer(),
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer()])
        else:
            self.conv1 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_width, 5, stride=1, padding=2, bias=False),
                    act_layer()])
        self.feature_info = [dict(num_chs=inplanes, reduction=0, module='act1')]

        # Feature Blocks
        channels = len(layers)*[128] # [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Modify Classification Head (Original ResNet) to Output Image Layer (Off-ResNet)
        self.conv_out = nn.Conv2d(inplanes, self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv_out(x)
        return x


class OffResNet2dBig(nn.Module):
    def __init__(
            self, block, layers, output_stride=32,
            cardinality=1, base_width=128, stem_width=128, stem_type='', block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., block_args=None, **kwargs):
        super(OffResNet2dBig, self).__init__()
        
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])
        
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width # stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width, stem_width)
            self.conv1 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer()])
            self.conv2 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer(),
                    nn.Conv2d(self.inp_channels, stem_chs[0], 5, stride=1, padding=2, bias=False),
                    act_layer()])
        else:
            self.conv1 = nn.Sequential(*[
                    nn.Conv2d(self.inp_channels, stem_width, 5, stride=1, padding=2, bias=False),
                    act_layer()])
        self.feature_info = [dict(num_chs=inplanes, reduction=0, module='act1')]

        # Feature Blocks
        channels = len(layers)*[128] # [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Modify Classification Head (Original ResNet) to Output Image Layer (Off-ResNet)
        self.conv_out = nn.Conv2d(inplanes, self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.conv_out(x)
        return x


@register_model
def offresnet2d(pretrained=False, **kwargs):
    """Constructs a Off-ResNet 2d model.
    """
    model = OffResNet2d(block=BasicBlock, layers=[3], args=kwargs['args'])
    model.default_cfg = _cfg()
    return model


@register_model
def offresnet2dbig(pretrained=False, **kwargs):
    """Constructs a Off-ResNet 2d model.
    """
    model = OffResNet2dBig(block=BasicBlock, layers=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], args=kwargs['args'])
    model.default_cfg = _cfg()
    return model