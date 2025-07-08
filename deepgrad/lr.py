import math

# --- Learning Rate Scheduler ---
class LinearDecayLR:
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = max(total_steps, 1)
        self.step_num = 0

    def step(self):
        pct = self.step_num / self.total_steps
        lr = self.start_lr + (self.end_lr - self.start_lr) * pct
        self.optimizer.set_lr(lr)
        self.step_num += 1

class OneCycleLR:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_num = 0

        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.phase1_steps = int(self.total_steps * self.pct_start)
        self.phase2_steps = self.total_steps - self.phase1_steps

    def _anneal_cos(self, start, end, pct):
        """Cosine annealing between start and end"""
        cos_out = (1 + math.cos(math.pi * pct)) / 2
        return end + (start - end) * cos_out

    def step(self):
        if self.step_num < self.phase1_steps:
            pct = self.step_num / self.phase1_steps
            lr = self._anneal_cos(self.initial_lr, self.max_lr, 1 - pct)
        else:
            pct = (self.step_num - self.phase1_steps) / max(1, self.phase2_steps)
            lr = self._anneal_cos(self.max_lr, self.final_lr, pct)

        self.optimizer.set_lr(lr)
        self.step_num += 1