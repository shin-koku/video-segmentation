import argparse


def opt():
    parser = argparse.ArgumentParser(description='options for seg_net')

    parser.add_argument('--directory',required=True,type=str,help='the directory of dataset')

    parser.add_argument('--annotation',required=True,type = str ,help=('the directory of annotation,regard as relative path to \'directory\' as default'))

    parser.add_argument('--output',required=True,type=str,help='the output path of model and results of network running')

    parser.add_argument('--resnext', default=101 ,type=int,help='the layers num of resnext,can be specified as [50,101,152]')

    parser.add_argument('--cardinality',default=32,type=int,help='the cardinality of resnext, for more information please view the paper')

    parser.add_argument('--epoches',default=50,type=int,help='epoches for training')

    parser.add_argument('--lr_decay',default=0.1,type=float ,help='the fator of lr decaying')

    parser.add_argument('--dampening',default=0 ,type = float,help = 'the dampening factor of SGD')

    parser.add_argument('--nesterov' ,action='store_true',type=bool,help='use nesterov momentum in SGD or not')

    parser.add_argument('--patience',default=10,type=int,help='the patience of lr_schedule which default use reducelronplateu')

    parser.add_argument('--num_classes',default=400,type = int ,help ='set the categories num of dataset')

    parser.add_argument('--no_train' ,action='store_false',type=bool,help='train or not')

    parser.add_argument('--no_validation', action='store_false',type=bool,help ='validation or not')

    return parser.parse_args()