import torch
import torch.nn as nn
import torch.optim as optim

from methods import BaseMethod, get_optimizer
from networks import to_device
from config import NUM_LOSSES
import losses


class GradNormMethod(BaseMethod):
    """
        The Gradient Normalization method for doing loss weighting.
        Paper: http://proceedings.mlr.press/v80/chen18a.html
    """

    def __init__(self, args, loader):
        super(GradNormMethod, self).__init__(args, loader)

        if args.mode == 'train':
            self.criterion = losses.BaseLoss(args)
            self.criterion = self.criterion.to(self.device)
            self.optimizer = get_optimizer(self.G.parameters(), args)

            # Parameters specific to GradNorm.
            self.params = torch.tensor([1.0 / NUM_LOSSES for _ in range(NUM_LOSSES)],
                                       requires_grad=True, device=self.device)

            self.first_iter = True
            self.L0 = None

            self.gamma = args.init_gamma
            self.Gradloss = nn.L1Loss()
            # Taking optimizer parameters from the GradNorm paper.
            self.optimizer_params = optim.Adam([self.params], lr=0.025)

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data['left_image']
        self.right = self.data['right_image']

    def forward(self):
        self.disps = self.G(self.left)

    def backward(self):
        # 0. Get the unweighted losses L.
        L = self.criterion(self.disps, [self.left, self.right])
        L = torch.stack(L)
        # for the first iteration with no l0.
        if self.first_iter:
            self.L0 = L.detach()
            self.first_iter = False

        # 1. Compute the training loss using the current set of weights.
        self.loss_G = torch.sum(torch.mul(self.params, L))
        # Moved here because of gradients: (6. Compute the standard gradients over the weights of the network.)
        self.optimizer.zero_grad()
        self.loss_G.backward(retain_graph=True)

        # 2. Compute norm of gradients G_{i} for each task i
        shared_weights = list(self.G.iconv4.parameters())
        # Getting gradients of the first layers of each task, with respect to the last shared layer.
        GR = [torch.autograd.grad(L[i], shared_weights[0], retain_graph=True) for i in range(NUM_LOSSES)]
        # The norm of the gradient w.r.t. wL is equivalent to w * the gradient w.r.t. L.
        G = [self.params[i] * torch.norm(GR[i][0], 2).reshape((1,)) for i in range(NUM_LOSSES)]
        G = torch.stack(G).view(-1)

        # 3. Compute the expected value of the gradients.
        G_avg = torch.mean(G)

        # 4. Compute the relative inverse rates.
        # The loss ratios concern the unweighted losses.
        lhat = torch.div(L, self.L0)
        lhat_avg = torch.mean(lhat)
        # Calculating relative inverse training rates for tasks
        inv_rate = torch.div(lhat, lhat_avg)

        # 5. Compute the gradient loss, keeping the targets constant. Thus use detach.
        C = (G_avg * inv_rate ** self.gamma).detach()

        # Set the gradients of params to zero, because we only want to update them using the gradient loss.
        self.optimizer_params.zero_grad()
        self.loss_params = torch.sum(self.Gradloss(G, C))
        self.params.grad = torch.autograd.grad(self.loss_params, self.params)[0]
        return self.loss_G.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        loss = self.backward()

        # 7. Update loss weightings
        self.optimizer_params.step()
        # 8. Update the network's weights
        self.optimizer.step()

        # 9. Renormalize the losses weights. Unlike the standard implementation, we normalize
        # such that the weights sum to 1, to fairly compare against our method. Do so without building a comp. graph.
        # Also, enforce loss weights never drop below 0.0.
        self.params.data = self.params.data.clamp_min_(0.0)
        coef = 1 / torch.sum(self.params)
        self.params.data = coef * self.params.data
        return loss

    def get_untrained_loss(self):
        L = self.criterion(self.disps, [self.left, self.right])
        with torch.no_grad():
            loss_G = sum([(self.params[i] * L[i]).item() for i in range(NUM_LOSSES)])
        return loss_G

    def store_val_loss(self, val_loss, epoch):
        super(GradNormMethod, self).store_val_loss(val_loss, epoch)
        # Besides also store the weights from past training epoch.
        if epoch >= 0:
            self.losses[epoch]['alphas'] = [self.params[i].detach().item() for i in range(NUM_LOSSES)]

    @property
    def architecture(self):
        return 'GradNorm Method'
