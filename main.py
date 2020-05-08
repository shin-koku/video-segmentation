import torch
import torch.nn as nn
import opt
import os
import train
import validation
import torch.optim as optim
from models import seg_resNext


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

def get_parameters(type,*args):
    return 0

if __name__ == '__main__':
    opts = opt.opt()
    work_directory = opts.directory
    annotation_directory  = os.path.join(work_directory,opts.annotation)
    output_path = os.path.join(work_directory,opts.output)


    resnext_model = generate_model(opts)
    parameters = get_parameters()
    optimizer = optim.SGD(parameters,lr=0.1,momentum=0.9,dampening=opts.dampening,nesterov=opts.nesterov)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=opts.patience,factor=opts.lr_decay)

    for i in range(opts.epoches):
        if not opts.no_train:





