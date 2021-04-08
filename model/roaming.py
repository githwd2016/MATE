# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: roaming.py
@Time: 2019/9/5 2:36 PM
@Description:
"""
import argparse
import os

from torch import nn, optim
import numpy as np
import torch


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, d_model, n_warmup_steps):
        # self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        # self._optimizer.step()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """Learning rate scheduling per step """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        print(lr)

        # for param_group in self._optimizer.param_groups:
        #     param_group['lr'] = lr


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    print(torch.cuda.get_device_name(1))
    print(torch.cuda.device_count())
