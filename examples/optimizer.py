import math

class SGD:
    def __init__(self, parameters, lr=0.001, clip_value=5.0):
        self.parameters = parameters
        self.lr = lr
        self.clip_value = clip_value

    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                data = param.data
                grad = param.grad
                for i, g in enumerate(grad):
                    if math.isfinite(g):
                        g = max(min(g, self.clip_value), -self.clip_value)
                        data[i] -= self.lr * g

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                for i in range(len(param.grad)):
                    param.grad[i] = 0.0
