import re
import importlib
import torch.optim as optim

from methods.base_method import BaseMethod


def find_method_using_name(method_name):
    # Given the option --method [METHOD],
    # the file "networks/{}_method.py" will be imported.
    method_name = re.sub('-', '', method_name)
    method_filename = "methods." + method_name + "_method"
    method_lib = importlib.import_module(method_filename)

    # In the file, the class called [MethodName]Method() will
    # be instantiated. It has to be a subclass of BaseMethod, and it is case-insensitive.
    method = None
    target_model_name = method_name.replace('_', '') + 'method'
    for name, cls in method_lib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseMethod):
            method = cls

    if method is None:
        error_string = "No method class with name {} was found in {}.py,".format(target_model_name, method_filename)
        raise ImportError(error_string)
    return method


def create_method(args, loader):
    model = find_method_using_name(args.method)
    instance = model(args, loader)
    print("Method {} with name {} was created".format(instance.__class__.__name__, instance.name))
    return instance


def get_optimizer(params, args):
    """ Gets and optimizer according to args.optimizer.
        For now use a set of hard-coded parameters for each optimizer.
    """
    which_optimizer = args.optimizer
    if which_optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.learning_rate)
    elif which_optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif which_optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError('{} has not been implemented'.format(which_optimizer))
    return optimizer
