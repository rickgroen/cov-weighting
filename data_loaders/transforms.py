import torch
import numpy as np


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class AugmentImage(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

        self.random_gamma = None
        self.random_brightness = None
        self.random_colours = None

    def set_random_variables(self):
        self.random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
        self.random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
        self.random_colours = np.random.uniform(self.color_low, self.color_high, 3)

    def __call__(self, sample):
        # Randomly shift gamma
        sample = sample ** self.random_gamma
        # Randomly shift brightness
        sample = sample * self.random_brightness
        # Randomly shift color
        for i in range(3):
            sample[i, :, :] *= self.random_colours[i]
        # Saturate
        return torch.clamp(sample, 0, 1)
