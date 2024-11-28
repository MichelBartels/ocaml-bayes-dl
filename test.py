import iree.runtime as ireert
import numpy as np
from mnist import mnist
from tqdm import tqdm
from matplotlib import pyplot as plt

train = ireert.load_vm_flatbuffer_file("train.vmfb", driver="local-task").fn320
decoder = ireert.load_vm_flatbuffer_file("decoder.vmfb", driver="local-task").fn1197

w1_encoder = np.random.normal(size=(784, 512)).astype(np.float32) * 0.01
b1_encoder = np.zeros((512,), dtype=np.float32)
w2_encoder = np.random.normal(size=(512, 16)).astype(np.float32) * 0.01
b2_encoder = np.zeros((16,), dtype=np.float32)
w3_encoder = np.random.normal(size=(512, 16)).astype(np.float32) * 0.01
b3_encoder = np.zeros((16,), dtype=np.float32)

w1_decoder = np.random.normal(size=(16, 512)).astype(np.float32) * 0.01
b1_decoder = np.zeros((512,), dtype=np.float32)
w2_decoder = np.random.normal(size=(512, 784)).astype(np.float32) * 0.01
b2_decoder = np.zeros((784,), dtype=np.float32)

params = [w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder, w1_decoder, b1_decoder, w2_decoder, b2_decoder]

num_epochs = 25000

bar = tqdm(zip(range(num_epochs), mnist()), total=num_epochs)

for i, (x, _) in bar:
    [*params, loss] = [x for x in train(*params, x)]
    bar.set_description(f"Loss: {loss.to_host()}")
    if (i + 1) % 2500 == 0:
        x = np.random.normal(size=(512, 16)).astype(np.float32)
        y_pred = decoder(*params[6:], x).to_host()[0]
        y_pred = y_pred.reshape(28, 28)
        plt.imshow(y_pred, cmap="gray")
        plt.show()
