import math

class SGD:
    def __init__(self, parameters, lr=0.001, clip_value=5.0):
        self.parameters = parameters
        self.lr = lr
        self.clip_value = clip_value

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                data = param.data
                grad = param.grad
                lr = self.lr
                clip_val = self.clip_value
                
                # Process in a single loop with fewer function calls
                for i, g in enumerate(grad):
                    if math.isfinite(g):
                        # Inline the clipping for better performance
                        g = clip_val if g > clip_val else (-clip_val if g < -clip_val else g)
                        data[i] -= lr * g

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                param.grad = [0.0] * len(param.grad)  # Faster than individual assignments