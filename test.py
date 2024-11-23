import iree.runtime as ireert
import numpy as np
from mnist import mnist

module = ireert.load_vm_flatbuffer_file("out.vmfb", driver="local-task")

w1 = np.random.normal(size=(784, 128)).astype(np.float32) * 0.01
b1 = np.zeros((128,), dtype=np.float32)

w2 = np.random.normal(size=(128, 10)).astype(np.float32) * 0.01
b2 = np.zeros((10,), dtype=np.float32)

for i, (x, y) in zip(range(10000), mnist()):
    w1, b1, w2, b2, loss = ([x.to_host() for x in module.fn69(w1, b1, w2, b2, y, x)])
    print(loss)
