"""
Learning Rate Scheduler Module

This module provides several learning rate schedulers compatible with optimizers
that implement a `.set_lr(new_lr)` method. Each scheduler defines a `.step()` method
to update the learning rate over training steps or epochs.

Available Schedulers:
- ConstantLR: Fixed learning rate.
- LinearDecayLR: Linearly decays from start_lr to end_lr over total_steps.
- CosineAnnealingLR: Cosine annealing from max_lr to min_lr over total_steps.
- StepLR: Drops LR by a factor every `step_size` steps.
- ExponentialLR: Decays LR exponentially each step by a factor `gamma`.
- OneCycleLR: Cosine ramp-up to max_lr then ramp-down to final_lr.

Example Usage:
--------------
from deepgrad.optim import Adam
from deepgrad.lr_scheduler import CosineAnnealingLR

optimizer = Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, max_lr=0.1, min_lr=0.001, total_steps=100)

for epoch in range(100):
    train(...)            # your training logic here
    scheduler.step()      # update the learning rate after each epoch
"""

import math
from abc import ABC, abstractmethod

# --- Base Scheduler ---
class LRScheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    @abstractmethod
    def get_lr(self):
        pass

    def step(self):
        lr = self.get_lr()
        self.optimizer.set_lr(lr)
        self.step_num += 1

# --- Constant LR ---
class ConstantLR(LRScheduler):
    def __init__(self, optimizer, lr):
        super().__init__(optimizer)
        self.lr = lr

    def get_lr(self):
        return self.lr

# --- Linear Decay LR ---
class LinearDecayLR(LRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = max(total_steps, 1)

    def get_lr(self):
        pct = self.step_num / self.total_steps
        return self.start_lr + (self.end_lr - self.start_lr) * pct

# --- Cosine Annealing LR ---
class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, total_steps):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = max(total_steps, 1)

    def get_lr(self):
        pct = self.step_num / self.total_steps
        cos_out = (1 + math.cos(math.pi * pct)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cos_out

# --- Step Decay LR ---
class StepLR(LRScheduler):
    def __init__(self, optimizer, initial_lr, drop_factor=0.1, step_size=10):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.step_size = step_size

    def get_lr(self):
        return self.initial_lr * (self.drop_factor ** (self.step_num // self.step_size))

# --- Exponential Decay LR ---
class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, initial_lr, gamma=0.9):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self):
        return self.initial_lr * (self.gamma ** self.step_num)

# --- One Cycle LR ---
class OneCycleLR(LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.phase1_steps = int(self.total_steps * self.pct_start)
        self.phase2_steps = self.total_steps - self.phase1_steps

    def _anneal_cos(self, start, end, pct):
        cos_out = (1 + math.cos(math.pi * pct)) / 2
        return end + (start - end) * cos_out

    def get_lr(self):
        if self.step_num < self.phase1_steps:
            pct = self.step_num / self.phase1_steps
            return self._anneal_cos(self.initial_lr, self.max_lr, 1 - pct)
        else:
            pct = (self.step_num - self.phase1_steps) / max(1, self.phase2_steps)
            return self._anneal_cos(self.max_lr, self.final_lr, pct)