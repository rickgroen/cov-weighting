import os
import torch
import pickle
from tqdm import tqdm

# project imports
from networks import define_G, to_device
from methods.scheduler import LearningRateScheduler


class BaseMethod:

    def __init__(self, args, loader):
        self.args = args
        self._name = args.model_name
        self.device = args.device
        self.mode = args.mode

        self.model_dir = args.model_dir
        self.model_store_path = os.path.join(args.model_dir, args.model_name)
        if not os.path.exists(self.model_store_path) and args.mode == 'train':
            os.mkdir(self.model_store_path)
            # Also store all args in a text_file.
            self.write_args_string_to_file()
        if not os.path.exists(self.model_store_path) and args.mode == 'test':
            raise FileNotFoundError('Model does not exist. Please check if the model has yet been run.')

        self.losses = {}
        self.loader = loader
        self.epochs = args.epochs

        # Define the generator network.
        self.G = define_G(args).to(self.device)
        print("[{}] initiated with {} trainable parameters".format(args.backbone, self.num_parameters))

        # Set the optimizer and scheduler, but wait for method-specific parameters.
        self.criterion = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        if args.mode == 'train':
            self.learning_rate_scheduler = LearningRateScheduler(args.adjust_lr, args.lr_mode, args.learning_rate,
                                                                 args.epochs, args.num_iterations)

    def predict(self, data):
        # Get the inputs
        data = to_device(data, self.args.device)
        return self.G(data)

    def run_epoch(self, current_epoch):
        tbar = tqdm(self.loader)
        total_iters = len(self.loader)
        train_loss = 0.0

        for current_iter, data in enumerate(tbar):
            # First, adjust learning rate.
            self.update_learning_rate(current_epoch, current_iter)
            # Then optimize.
            self.set_input(data)
            iter_loss = self.optimize_parameters()
            # Gather running loss, so we can compute the full loss after the epoch.
            train_loss += iter_loss
            tbar.set_description('Train loss: %.6f' % (train_loss / (current_iter + 1)))
        # Record the running loss.
        if current_epoch not in self.losses:
            self.losses[current_epoch] = {}
        self.losses[current_epoch]['train'] = train_loss / total_iters

    def set_input(self, data):
        pass

    def optimize_parameters(self):
        pass

    def update_learning_rate(self, current_epoch, current_iter):
        self.learning_rate_scheduler(self.optimizer, current_epoch, current_iter)

    def get_untrained_loss(self):
        pass

    def store_val_loss(self, val_loss, epoch):
        if epoch not in self.losses:
            self.losses[epoch] = {}
        # Set the validation loss.
        self.losses[epoch]['val'] = val_loss

    def save_network(self, name):
        save_filename = "{}_G.pth".format(name)
        save_path = os.path.join(self.model_store_path, save_filename)
        torch.save(self.G.state_dict(), save_path)

    def load_network(self, name):
        load_filename = "{}_G.pth".format(name)
        load_path = os.path.join(self.model_store_path, load_filename)
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.G.load_state_dict(state_dict)

    def save_losses(self):
        path = os.path.join(self.model_store_path, 'losses.p'.format(self._name))
        with open(path, 'wb') as loss_output_file:
            pickle.dump(self.losses, loss_output_file)

    def to_eval(self):
        self.G.eval()
        if self.mode == 'train':
            self.criterion.to_eval()

    def to_train(self):
        self.G.train()
        self.criterion.to_train()

    def write_args_string_to_file(self):
        args_string = '=== ARGUMENTS ===\n'
        for key, val in vars(self.args).items():
            args_string += '{0: <20}: {1}\n'.format(key, val)
        args_string += '=================\n'
        with open(os.path.join(self.model_store_path, 'params'), 'w') as f:
            f.write(args_string)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.G.parameters() if p.requires_grad)

    @property
    def method(self):
        return 'Base Method'

    @property
    def name(self):
        return self._name
