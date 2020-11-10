import collections

from networks.blocks import *
from networks.resnet import ResNet18MD, ResNet50MD


def define_G(args):
    if args.norm_layer == 'batch':
        normalize = nn.BatchNorm2d
    elif args.norm_layer == 'instance':
        normalize = nn.InstanceNorm2d
    else:
        normalize = None

    # Method specfic configurations.
    if args.backbone[:6] != 'resnet' or int(args.backbone[6:]) not in [18, 50]:
        raise NotImplementedError("Only implemented Resnet 18 and 50 generators for now.")
    resnet_layers = int(args.backbone[6:])
    return_shared_layer = 'multi-objective' in args.method

    if resnet_layers == 18:
        model = ResNet18MD(normalize=normalize, do_multi_objective=return_shared_layer)
    else:
        model = ResNet50MD(normalize=normalize, do_multi_objective=return_shared_layer)
    print('Now initializing {} generator using {} normalization'.format(model.__class__.__name__,
                                                                        'no' if not normalize else args.norm_layer))
    return model


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")
