from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.MNIST('data/', download=True, transform=transform)

batch_size = 32

def cycle(loader):
    while True:
        for x in loader:
            yield x

def mnist():
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for x, y in cycle(loader):
        x = x.reshape(x.shape[0], -1)
        y = one_hot(y, 10)
        yield x.numpy().astype(np.float32), y.numpy().astype(np.float32)
