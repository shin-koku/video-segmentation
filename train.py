from torch.autograd import Variable
import torch
import torch.nn


def flatten_for_LSTM():
    pass

def train_epoch(epoches, dataset, optimizer, classification, resnext, conv_block, LSTM, logger_pth, opts):
    for i, (inputs, targets) in enumerate(dataset):
        if not opts.no_cuda:
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        res_out = resnext(inputs)
        conv_out = list(ind_conv_block(res_out) for ind_conv_block in conv_block)
        out = []
        for lstm in LSTM:
            out.append(lstm(conv_out))
        loss = list(classification(lstm_out) for lstm_out in out)
        for _loss in loss:
            _loss.backward()
            optimizer.step()
