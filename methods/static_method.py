from methods import BaseMethod, get_optimizer
from networks import to_device
import losses


class StaticMethod(BaseMethod):

    def __init__(self, args, loader):
        super(StaticMethod, self).__init__(args, loader)

        if args.mode == 'train':
            self.criterion = losses.StaticLoss(args)
            self.criterion = self.criterion.to(self.device)
            self.optimizer = get_optimizer(self.G.parameters(), args)

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data['left_image']
        self.right = self.data['right_image']

    def forward(self):
        self.disps = self.G(self.left)

    def backward(self):
        loss_G = self.criterion(self.disps, [self.left, self.right])
        loss_G.backward()
        return loss_G.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        loss = self.backward()
        self.optimizer.step()
        return loss

    def get_untrained_loss(self):
        loss_G = self.criterion(self.disps, [self.left, self.right])
        return loss_G.item()

    @property
    def architecture(self):
        return 'Static Method'
