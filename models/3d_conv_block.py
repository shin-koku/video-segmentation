import torch
import torch.nn as nn
import enum

__all__ = {'scale_4_block','scale_8_block','scale_16_block'}



def scale_4_block(in_size,cardinality):
    return block_builder(4,1,in_size,cardinality)



def scale_8_block(in_size,cardinality):
    return block_builder(8,2,in_size,cardinality)



def scale_16_block(in_size,cardinality):
    return block_builder(16,3,in_size,cardinality)


def block_builder(scale,num,in_size,cardinality):
    block = []
    for i in range(num):
        block.append(basic_conv_block(in_size,scale,num,cardinality))
    return nn.Sequential(*block)

class basic_conv_block(nn.Sequential):

    def __init__(self,in_size,scale,num,cardinality):
        super(basic_conv_block,self).__init__()
        mid_size = cardinality * int(in_size / 32)
        self.add_module('conv1_{}_{}'.format(scale,num),nn.Conv3d(
            in_size,
            mid_size,
            kernel_size=1,
            stride=1,
            bias=False
        ))
        self.add_module('relu1'.format(scale,num),nn.ReLU(inplace=True))
        self.add_module('bn1_{}_{}'.format(scale,num),nn.BatchNorm3d(mid_size))
        self.add_module('conv2_{}_{}'.format(scale,num).format(scale,num),nn.Conv3d(
            mid_size,
            mid_size,
            kernel_size=3,
            stride=1,
            groups= cardinality,
            padding=1,
            bias=False
        ))
        self.add_module('relu2'.format(scale,num), nn.ReLU(inplace=True))
        self.add_module('bn1_{}_{}'.format(scale,num), nn.BatchNorm3d(mid_size))
        self.add_module('conv3_{}_{}'.format(scale,num).format(scale, num), nn.Conv3d(
            mid_size,
            1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))
        self.add_module('relu2'.format(scale,num), nn.ReLU(inplace=True))
        self.add_module('bn1_{}_{}'.format(scale,num), nn.BatchNorm3d(1024))
        self.add_module('avg_pool',nn.AvgPool3d(kernel_size=(2,1,1),stride=2))
