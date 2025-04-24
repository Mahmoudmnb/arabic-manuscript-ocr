import torch
from doctr.transforms import GaussianNoise
transfo = GaussianNoise(0., 1.)
out = transfo(torch.rand((3, 224, 224)))
print(out)