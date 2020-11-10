import torch

from .base_loss import BaseLoss


class StaticLoss(BaseLoss):

    """
        Wrapper of the BaseLoss which weights the losses according to static weights.
    """

    def __init__(self, args):
        super(StaticLoss, self).__init__(args)
        self.device = args.device

        self.l1_w = args.img_loss_l1_w
        self.ssim_w = args.img_loss_ssim_w
        self.lr_w = args.lr_loss_w
        self.disp_gradient_w = args.disp_grad_loss_w

        # Store the static weights, so they can be recorded.
        self.alphas = torch.tensor([self.l1_w / 8, self.l1_w / 8, self.l1_w / 8, self.l1_w / 8, self.l1_w / 8, self.l1_w / 8, self.l1_w / 8, self.l1_w / 8,
                                    self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8, self.ssim_w / 8,
                                    self.lr_w / 8, self.lr_w / 8, self.lr_w / 8, self.lr_w / 8, self.lr_w / 8, self.lr_w / 8, self.lr_w / 8, self.lr_w / 8,
                                    self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8, self.disp_gradient_w / 8],
                                   requires_grad=False).type(torch.FloatTensor).to(self.device)
        assert (torch.sum(self.alphas) - 1.0) < 1e-3, "Alphas do not sum to 1."

    def forward(self, pred, target):
        unweighted_losses = super(StaticLoss, self).forward(pred, target)
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        return sum(weighted_losses)
