import argparse
import ast
import os

HOME = os.environ['HOME']


def boolstr(s):
    """ Defines a boolean string, which can be used for argparse.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class BaseOptions:

    def __init__(self):
        pass

    def get_arguments(self, parser):
        parser.add_argument('--method',             type=str,   help='which method [static|cov-weighting|uncertainty|gradnorm|multi-objective] to run', default='cov-weighting')
        parser.add_argument('--dataset',            type=str,   help='dataset to train on kitti or cityscapes or both', default='kitti')

        parser.add_argument('--data_dir',           type=str,   help='path to the data directory', required=True)
        parser.add_argument('--model_dir',          type=str,   help='path to the trained models', default='saved_models/')
        parser.add_argument('--model_name',         type=str,   help='model name', default='cov-weighting-test')

        parser.add_argument('--input_height',       type=int,   help='input height', default=256)
        parser.add_argument('--input_width',        type=int,   help='input width', default=512)
        parser.add_argument('--backbone',           type=str,   help='backbone architecture with amount of layers', default='resnet50')

        parser.add_argument('--device',             type=str,   help='choose cpu or cuda:0 device"', default='cuda:0')
        parser.add_argument('--num_workers',        type=int,   help='number of threads to use for data loading', default=16)

        return parser

    @staticmethod
    def print_options(args):
        print('=== ARGUMENTS ===')
        for key, val in vars(args).items():
            print('{0: <20}: {1}'.format(key, val))
        print('=================')

    def parse(self):
        parser = argparse.ArgumentParser(description='CoV-Weighting PyTorch Implementation')
        parser = self.get_arguments(parser)

        args = parser.parse_args()

        # If the user has selected the home directory e.g. by --data_dir ~/data
        # then we need to make sure we get the absolute path for the PIL library.
        # Also obtain the correct data folder for each set.
        if args.data_dir[0] == '~':
            args.data_dir = os.path.join(HOME, args.data_dir[2:])
        args.data_dir = os.path.join(args.data_dir, args.dataset)

        # Convert certain items to lists.
        if hasattr(args, 'augment_parameters') and not type(args.augment_parameters) == list:
            args.augment_parameters = ast.literal_eval(args.augment_parameters)

        # Print the options.
        self.print_options(args)

        return args
