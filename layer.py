import torch

class DropBlock(torch.nn.module):
    def __init__(self, block_size=7, keep_prob=0.9):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)
    
    def calculate_gamma(self, x):
        return (1 - self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 
    
    def forward(self, x):
        if not self.training:
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        p = torch.ones_like(x) * (1 - self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)
        return mask * x * (mask.numel()/mask.sum())
