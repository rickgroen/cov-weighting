import torch
from torch import nn

from .base_loss import BaseLoss


class UncertaintyLoss(BaseLoss):

    """
        Wrapper of the BaseLoss which weighs the losses according to the Uncertainty Weighting
        method by Kendall et al. Now for 32 losses.
    """

    def __init__(self, args):
        super(UncertaintyLoss, self).__init__(args)
        # These are the log(sigma ** 2) parameters.
        self.log_vars = nn.Parameter(torch.zeros(self.num_losses, dtype=torch.float32), requires_grad=True)

    def to_eval(self):
        self.alphas = torch.exp(-self.log_vars).detach().clone()
        self.train = False

    def to_train(self):
        self.train = True

    def forward(self, pred, target):
        unweighted_losses = super(UncertaintyLoss, self).forward(pred, target)
        # -- Kendall's Gaussian method: L_total = sum_i exp(-s_i) * L_i + s_i --
        losses = [torch.exp(-self.log_vars[i]) * loss + self.log_vars[i] for i, loss in enumerate(unweighted_losses)]
        loss = sum(losses)
        return loss
