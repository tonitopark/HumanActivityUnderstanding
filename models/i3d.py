import numpy as np

import torch


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpt(torch.nn.Module):
    """Basic unit containing Conv3D + BatchNorm + Non-linearity."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpt, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias
                )
            else:  # not same length padding
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=pad_size,
                    stride=stride,
                    bias=use_bias)
        elif padding == 'VALID':  # VALID
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)

        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

            if activation == 'relu':
                self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = self.activation(out)
        return (out)


class MaxPool3DTFPadding(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3DTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return (out)


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()

        # Branch 0
        self.branch_0 = Unit3Dpt(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1

        branch_1_conv1 = Unit3Dpt(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpt(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2

        branch_2_conv1 = Unit3Dpt(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpt(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch 3

        branch_3_pool = MaxPool3DTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpt(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    'batch_size` x `num_frames` x 224 x 224 x `num_channels`.'

    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        self.conv3d_1a_7x7 = Unit3Dpt(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')

        self.maxpool3d_2a_3x3 = MaxPool3DTFPadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding='SAME')

        self.conv3d_2b_1x1 = Unit3Dpt(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')

        self.conv3d_2c_3x3 = Unit3Dpt(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.maxpool3d_3a_3x3 = MaxPool3DTFPadding(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding='SAME')

        self.mixed_3b = Mixed(
            in_channels=192,
            out_channels=[64, 96, 128, 16, 32, 32])

        self.mixed_3c = Mixed(
            in_channels=256,
            out_channels=[128, 128, 192, 32, 96, 64])

        self.maxpool3d_4a_3x3 = MaxPool3DTFPadding(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding='SAME')

        self.mixed_4b = Mixed(
            in_channels=480,
            out_channels=[192, 96, 208, 16, 48, 64])

        self.mixed_4c = Mixed(
            in_channels=512,
            out_channels=[160, 112, 224, 24, 64, 64])

        self.mixed_4d = Mixed(
            in_channels=512,
            out_channels=[128, 128, 256, 24, 64, 64])

        self.mixed_4e = Mixed(
            in_channels=512,
            out_channels=[112, 144, 288, 32, 64, 64])

        self.mixed_4f = Mixed(
            in_channels=528,
            out_channels=[256, 160, 320, 32, 128, 128])

        self.maxpool3d_5a_2x2 = MaxPool3DTFPadding(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding='SAME')

        self.mixed_5b = Mixed(
            in_channels=832,
            out_channels=[256, 160, 320, 32, 128, 128])

        self.mixed_5c = Mixed(
            in_channels=832,
            out_channels=[384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d(
            kernel_size=(2, 7, 7),
            stride=(1, 1, 1))

        self.dropout = torch.nn.Dropout(dropout_prob)

        self.conv3d_0c_1x1 = Unit3Dpt(
            in_channels=1024,
            out_channels=num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        out = self.conv3d_1a_7x7(inp)
        out = self.maxpool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxpool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxpool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxpool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)  # logits(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out_logits = out.mean(2)
        out = self.softmax(out_logits)

        return out, out_logits


if __name__ == "__main__":
    net = I3D(num_classes=400)

    ten = torch.randn((1, 3, 100, 224, 224))

    res = net(ten)

    print(res)
