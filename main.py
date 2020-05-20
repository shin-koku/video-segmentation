import os
import torch.optim as optim
import torch.nn as nn
import torch
import opt
import math
import GTEA
from models._conv_block import scale_block
from models import seg_resNext
from train import train_epoch
from validation import validation_epoch
from LSTM_part import seg_LSTM


def generate_model(opts):
    assert (opts.resnext in [50, 101, 152])
    if opt.resnext == 50:
        resnext_model = seg_resNext.resnet50(
            opts.cardinality, opts.num_cat
        )
    if opt.resnext == 101:
        resnext_model = seg_resNext.resnet101(
            opts.cardinality, opts.num_cat
        )
    if opt.resnext == 152:
        resnext_model = seg_resNext.resnet5152(
            opts.cardinality, opts.num_cat
        )


def LSTM(input_size, hidden_size, timestep, num_layers, bi_directional):
    assert len(timestep) == 4
    assert (timestep[i] * 2 == timestep[i + 1] for i in range(len(timestep) - 2))
    lstm = []
    lstm.append(
        seg_LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, bi_directional=bi_directional) for i in
        range(len(len) - 1))
    return lstm


def get_parameters(*args):
    return nn.Parameter()


if __name__ == '__main__':
    opts = opt.opt()
    scale = [4, 8, 16]
    work_directory = opts.directory
    annotation_directory = os.path.join(work_directory, opts.annotation)
    output_path = os.path.join(work_directory, opts.output)

    resnext_model = generate_model(opts)
    conv_block = []
    conv_block.append(scale_block(_, 1024, opts.cardinality) for _ in scale)
    LSTM_part = LSTM(1024, opts.time_steps, opts.num_layers, opts.bi)
    parameters = get_parameters()
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9, dampening=opts.dampening, nesterov=opts.nesterov)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.patience, factor=opts.lr_decay)

    classification = nn.CrossEntropyLoss()
    if not opts.no_cuda:
        classification = classification.cuda()

    for i in range(opts.epoches):
        if not opts.no_train:
            train_epoch(i, optimizer, GTEA.generate_dataset,classification, resnext_model,
                        conv_block, LSTM_part, output_path, opts)

        if not opts.no_validation:
            validation_epoch()
