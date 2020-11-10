import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

from methods import BaseMethod, get_optimizer
from networks import to_device
import losses


class MultiObjectiveMethod(BaseMethod):

    """
        Multi-objective Optimization for multi-loss weighting.
        Paper: http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization
        Code: https://github.com/intel-isl/MultiObjectiveOptimization
    """

    def __init__(self, args, loader):
        super(MultiObjectiveMethod, self).__init__(args, loader)

        if args.mode == 'train':
            self.criterion = losses.BaseLoss(args)
            self.criterion = self.criterion.to(self.device)
            self.optimizer = get_optimizer(self.G.parameters(), args)

            # Record the mean weights for an epoch.
            self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data['left_image']
        self.right = self.data['right_image']

    def forward(self):
        # If we are doing a validation pass.
        if not self.G.training:
            _, self.disps = self.G(self.left)
            return

        # Else get the no the shared representation.
        with torch.no_grad():
            (self.skip2, self.skip1, self.shared), _ = self.G(self.left)

    def backward(self):
        # Compute gradients of each loss function wrt z
        grads = []
        shared_layer = Variable(self.shared.data.clone(), requires_grad=True)

        disp4 = self.G.disp4_layer(shared_layer)
        udisp4 = nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.G.upconv3(shared_layer)
        concat3 = torch.cat((upconv3, self.skip2, udisp4), 1)
        iconv3 = self.G.iconv3(concat3)
        disp3 = self.G.disp3_layer(iconv3)
        udisp3 = nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.G.upconv2(iconv3)
        concat2 = torch.cat((upconv2, self.skip1, udisp3), 1)
        iconv2 = self.G.iconv2(concat2)
        disp2 = self.G.disp2_layer(iconv2)
        udisp2 = nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.G.upconv1(iconv2)
        concat1 = torch.cat((upconv1, udisp2), 1)
        iconv1 = self.G.iconv1(concat1)
        disp1 = self.G.disp1_layer(iconv1)

        # And get the losses.
        losses = self.criterion((disp1, disp2, disp3, disp4), [self.left, self.right])
        for ind, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            grads.append(Variable(shared_layer.grad.data.clone(), requires_grad=False))
            shared_layer.grad.data.zero_()

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[ind] for ind, _ in enumerate(losses)])
        scales = list(sol)

        # Scaled back-propagation
        self.optimizer.zero_grad()
        _, self.disps = self.G(self.left)
        losses = self.criterion(self.disps, [self.left, self.right])
        scaled_losses = [scale * loss for scale, loss in zip(scales, losses)]

        loss_G = sum(scaled_losses)
        loss_G.backward()

        # Finally, add the scales to the mean scales to get an idea of the mean scales after training.
        for i, scale in enumerate(scales):
            self.mean_weights[i] += (scale / len(self.loader))
        return loss_G.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        loss = self.backward()
        self.optimizer.step()
        return loss

    def get_untrained_loss(self):
        losses = self.criterion.forward(self.disps, [self.left, self.right])
        loss_G = sum(losses)
        return loss_G.item()

    def store_val_loss(self, val_loss, epoch):
        super(MultiObjectiveMethod, self).store_val_loss(val_loss, epoch)
        # Besides also store the weights from past training epoch.
        if epoch >= 0:
            self.losses[epoch]['alphas'] = copy.deepcopy(self.mean_weights)
            self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    @property
    def method(self):
        return 'Multi-Objective Method'


class MinNormSolver:

    """
        MinNorm Solver from 'Multi-Task Learning as Multi-Objective Optimization'
        by Ozan Sener, Vladlen Koltun
        published in Neural Information Processing Systems (NeurIPS) 2018
        github: https://github.com/intel-isl/MultiObjectiveOptimization
    """

    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k], vecs[j][k]).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k], vecs[i][k]).item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        # Reshape to get 1D vectors so that we can do dot products.
        vecs = [tens[0].reshape(1, tens[0].numel()) for tens in vecs]
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
