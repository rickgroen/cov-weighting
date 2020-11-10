import torch

from methods import BaseMethod, get_optimizer
from networks import to_device
import losses
from config import NUM_LOSSES


class UncertaintyMethod(BaseMethod):

    """
        Uncertainty Weighting Method.
        Paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, args, loader):
        super(UncertaintyMethod, self).__init__(args, loader)

        if args.mode == 'train':
            self.criterion = losses.UncertaintyLoss(args)
            self.criterion.to(self.device)

            # Specifically for this architecture, both model parameters and the estimates
            # of the log variances have to be optimized.
            params_to_optimize = list(self.G.parameters()) + [self.criterion.log_vars]
            self.optimizer = get_optimizer(params_to_optimize, args)

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
        return loss

    def get_untrained_loss(self):
        loss_G = self.criterion.forward(self.disps, [self.left, self.right])
        return loss_G.item()

    def store_val_loss(self, val_loss, epoch):
        super(UncertaintyMethod, self).store_val_loss(val_loss, epoch)
        # Besides also store the weights from past training epoch.
        if epoch >= 0:
            self.losses[epoch]['alphas'] = [torch.exp(-self.criterion.log_vars[i]).detach().item()
                                            for i in range(NUM_LOSSES)]

    @property
    def architecture(self):
        return 'Uncertainty Method'
