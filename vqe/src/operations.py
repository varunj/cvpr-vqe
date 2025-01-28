import torch.nn as nn
from enum import Enum

EPS_BN = 1e-3


def make_batch_norm(channels, eps=EPS_BN, momentum=0.1):
    # pytorch default eps 1e-5
    # tensorflow default eps 1e-3, and it seems to be more stable (no NaNs)
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)


class Activation(Enum):
    SIGMOID = 'sigmoid'
    CLIP = 'clip'
    RELU = 'relu'
    RELU1 = 'relu1'
    RELU6 = 'relu6'
    PRELU = 'prelu'
    LEAKY_RELU = 'leaky_relu'
    NONE = 'none'


def make_activation(act_string: str):
    act = Activation('none' if act_string is None else act_string)
    if act == Activation.SIGMOID:
        return nn.Sigmoid()
    elif act == Activation.CLIP:
        return nn.Hardtanh(0.0, 1.0, inplace=True)
    elif act == Activation.RELU:
        return nn.ReLU(inplace=True)
    elif act == Activation.RELU1:
        return nn.Hardtanh(-1.0, 1.0, inplace=True)
    elif act == Activation.RELU6:
        return nn.ReLU6(inplace=True)
    elif act == Activation.PRELU:
        return nn.PReLU()
    elif act == Activation.LEAKY_RELU:
        return nn.LeakyReLU(inplace=True)
    elif act == Activation.NONE:
        return nn.Identity()


class Downsample(Enum):
    AVGPOOL = 'avgpool'
    MAXPOOL = 'maxpool'


def make_downsample(d_string: str):
    ds = Downsample(d_string)
    if ds == Downsample.AVGPOOL:
        return nn.AvgPool2d(2, 2, count_include_pad=False)
    elif ds == Downsample.MAXPOOL:
        return nn.MaxPool2d(2, 2)


class OverParametrized1x1(nn.Module):
    def __init__(self, channels_in, channels_out, activation, over_parametrize, invert_bn_act=False, is_last=False):
        super().__init__()
        self._invert_bn_act = invert_bn_act
        self._is_last = is_last
        self._activation = make_activation(activation)
        self._ops = nn.ModuleList()
        self._ops.append(
            nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, bias=False),
                nn.Identity() if self._invert_bn_act else make_batch_norm(channels_out)
            )
        )
        if over_parametrize > 0 and channels_in == channels_out and not self._invert_bn_act:
            self._ops.append(make_batch_norm(channels_out))
        for _ in range(1, over_parametrize):
            self._ops.append(
                nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, 1, bias=False),
                    nn.Identity() if self._invert_bn_act else make_batch_norm(channels_out)
                )
            )
        if self._invert_bn_act:
            self._bn = make_batch_norm(channels_out)

    def forward(self, x):
        if len(self._ops) > 1:
            y = sum([op(x) for op in self._ops])
        else:
            y = self._ops[0](x)

        if not self._invert_bn_act:
            return self._activation(y)
        else:
            if self._activation.__class__ ==  nn.Sigmoid().__class__ and self._is_last:
                return self._activation(self._bn(y))
            else:
                return self._bn(self._activation(y))


class OverParametrized3x3dw(nn.Module):
    def __init__(self, channels, activation, over_parametrize, invert_bn_act=False):
        super().__init__()
        self._invert_bn_act = invert_bn_act
        self._activation = make_activation(activation)
        self._ops = nn.ModuleList()
        self._ops.append(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, groups=channels, padding=1, bias=False),
                nn.Identity() if self._invert_bn_act else make_batch_norm(channels)
            )
        )
        if over_parametrize > 0:
            if not self._invert_bn_act:
                self._ops.append(make_batch_norm(channels))
            self._ops.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
                    nn.Identity() if self._invert_bn_act else make_batch_norm(channels)
                )
            )

        for _ in range(1, over_parametrize):
            self._ops.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, groups=channels, padding=1, bias=False),
                    nn.Identity() if self._invert_bn_act else make_batch_norm(channels)
                )
            )
        if self._invert_bn_act:
            self._bn = make_batch_norm(channels)

    def forward(self, x):
        if len(self._ops) > 1:
            y = sum([op(x) for op in self._ops])
        else:
            y = self._ops[0](x)

        if not self._invert_bn_act:
            return self._activation(y)
        else:
            return self._bn(self._activation(y))


class SEBlock(nn.Module):
    def __init__(self, channels, hidden_channels, mid_activation='relu',
        batch_norm=True, final_batch_norm=True, invert_bn_act=False, scale_activation=None):
        super(SEBlock, self).__init__()
        self._squeeze = nn.AdaptiveAvgPool2d(1)
        if batch_norm:
            if invert_bn_act:
                self._excitation = nn.Sequential(
                    nn.Conv2d(channels, hidden_channels, 1, bias=False),
                    make_activation(mid_activation),
                    make_batch_norm(hidden_channels),
                    nn.Conv2d(hidden_channels, channels, 1, bias=False),
                )
            else:
                self._excitation = nn.Sequential(
                    nn.Conv2d(channels, hidden_channels, 1, bias=False),
                    make_batch_norm(hidden_channels),
                    make_activation(mid_activation),
                    nn.Conv2d(hidden_channels, channels, 1, bias=False),
                )
        else:
            self._excitation = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, 1, bias=True),
                make_activation(mid_activation),
                nn.Conv2d(hidden_channels, channels, 1, bias=False),
            )
        self._scale_activation = make_activation(scale_activation)
        if final_batch_norm:
            self._final = make_batch_norm(channels)
        else:
            self._final = nn.Identity()

    def forward(self, x):
        squeezed = self._squeeze(x)
        excite = self._excitation(squeezed)
        excite = self._scale_activation(excite)
        return self._final(x * excite)


def make_m1_block(channels_in, channels_out, activation, over_parametrize, invert_bn_act=False):
    return nn.Sequential(
        OverParametrized1x1(channels_in, channels_out, activation, over_parametrize, invert_bn_act=invert_bn_act),
        OverParametrized3x3dw(channels_out, activation, over_parametrize, invert_bn_act=invert_bn_act)
    )


def make_block_list(channels_in, channel_list, activation, over_parametrize, se_block=None, invert_bn_act=False):
    if len(channel_list) == 0:
        return nn.Identity()
    block_list = []
    ch_in = channels_in
    for ch_out in channel_list:
        block_list.append(make_m1_block(ch_in, ch_out, activation, over_parametrize, invert_bn_act=invert_bn_act))
        ch_in = ch_out
    if se_block is not None:
        block_list.append(
            SEBlock(
                ch_in, se_block['hidden'],
                mid_activation=se_block['mid_activation'],
                batch_norm=se_block['bn'],
                final_batch_norm=se_block['final_bn'],
                invert_bn_act=invert_bn_act,
                scale_activation=se_block['scale_activation'] if 'scale_activation' in se_block else None,
            )
        )
    return nn.Sequential(*block_list)
