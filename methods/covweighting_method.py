import copy

from methods import BaseMethod, get_optimizer
from networks import to_device
import losses


class CoVWeightingMethod(BaseMethod):

    def __init__(self, args, loader):
        super(CoVWeightingMethod, self).__init__(args, loader)

        if args.mode == 'train':
            self.criterion = losses.CoVWeightingLoss(args)
            self.criterion.to(self.device)
            self.optimizer = get_optimizer(self.G.parameters(), args)

            # Record the mean weights for an epoch.
            self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data['left_image']
        self.right = self.data['right_image']

    def forward(self):
        self.disps = self.G(self.left)

    def backward(self):
        loss_G = self.criterion.forward(self.disps, [self.left, self.right])
        loss_G.backward()
        return loss_G.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        loss = self.backward()
        self.optimizer.step()

        # Finally, add the scales to the mean scales to get an idea of the mean weights after training.
        for i, weight in enumerate(self.criterion.alphas):
            self.mean_weights[i] += (weight.item() / len(self.loader))
        return loss

    def get_untrained_loss(self):
        loss_G = self.criterion.forward(self.disps, [self.left, self.right])
        return loss_G.item()

    def store_val_loss(self, val_loss, epoch):
        super(CoVWeightingMethod, self).store_val_loss(val_loss, epoch)
        # Besides also store the weights from past training epoch.
        if epoch >= 0:
            self.losses[epoch]['alphas'] = copy.deepcopy(self.mean_weights)
            self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    @property
    def method(self):
        return 'CoV-Weighting Method'
