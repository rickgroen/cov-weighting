import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.bilinear_sampler import apply_disparity
from config import NUM_LOSSES


class BaseLoss(nn.modules.Module):

    def __init__(self, args):
        super(BaseLoss, self).__init__()
        self.device = args.device

        self.n = 4
        self.train = False

        # Record the weights.
        self.num_losses = NUM_LOSSES
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

    @staticmethod
    def scale_pyramid(img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img, [nh, nw], mode='bilinear', align_corners=False))
        return scaled_imgs

    @staticmethod
    def gradient_x(img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    @staticmethod
    def gradient_y(img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    @staticmethod
    def generate_image_left(img, disp):
        return apply_disparity(img, -disp)

    @staticmethod
    def generate_image_right(img, disp):
        return apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.functional.avg_pool2d(x, 3, 1, padding=0)
        mu_y = nn.functional.avg_pool2d(y, 3, 1, padding=0)

        sigma_x = nn.functional.avg_pool2d(x ** 2, 3, 1, padding=0) - mu_x ** 2
        sigma_y = nn.functional.avg_pool2d(y ** 2, 3, 1, padding=0) - mu_y ** 2

        sigma_xy = nn.functional.avg_pool2d(x * y, 3, 1, padding=0) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return smoothness_x + smoothness_y

    def to_eval(self):
        self.train = False

    def to_train(self):
        self.train = True

    def forward(self, pred, target):
        """ Computes the basic loss functions, without combining the components
            into the full objective yet. The loss wrappers take care of those.
            The predictions can either be a tuple of multi-scaled disps (as in Godard's MonoDepth)
            or a single disparity.
            According to our paper 'On the Benefit of Adversarial Learning for Monocular Depth Estimation'
            performance using a single full-scale disparity yields slightly better performance.
        Args:
            pred    [disp1, disp2, disp3, disp4] OR disp
            target  [left, right]

        Return:
            (float): The loss
        """
        left, right = target

        # Obtain the left, right disparities.
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in pred]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in pred]
        # Apply scaling to the targets.
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i], disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i], disp_right_est[i]) for i in range(self.n)]

        # L1
        l1_left_loss = [torch.mean(torch.abs(left_est[i] - left_pyramid[i])) for i in range(self.n)]
        l1_right_loss = [torch.mean(torch.abs(right_est[i] - right_pyramid[i])) for i in range(self.n)]
        l1_loss = l1_left_loss + l1_right_loss

        # SSIM
        ssim_left_loss = [torch.mean(self.SSIM(left_est[i], left_pyramid[i])) for i in range(self.n)]
        ssim_right_loss = [torch.mean(self.SSIM(right_est[i], right_pyramid[i])) for i in range(self.n)]
        ssim_loss = ssim_left_loss + ssim_right_loss

        # Left-Right Consistency (LR)
        right_left_disp = [self.generate_image_left(disp_right_est[i], disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i], disp_right_est[i]) for i in range(self.n)]

        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i])) for i in range(self.n)]
        lr_loss = lr_left_loss + lr_right_loss

        # Disparities smoothness (DISP)
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        disp_left_loss_x = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_left_loss_y = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** (i - self.n)
                            for i in range(self.n, self.n * 2)]
        disp_left_loss = [disp_left_loss_x[i] + disp_left_loss_y[i] for i in range(self.n)]

        disp_right_loss_x = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss_y = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** (i - self.n)
                             for i in range(self.n, self.n * 2)]
        disp_right_loss = [disp_right_loss_x[i] + disp_right_loss_y[i] for i in range(self.n)]

        disp_loss = disp_left_loss + disp_right_loss

        # Return all 32 losses as a list.
        loss = l1_loss + ssim_loss + lr_loss + disp_loss
        return loss
