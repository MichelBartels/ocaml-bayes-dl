import iree.runtime as ireert
import numpy as np
from mnist import mnist
from tqdm import tqdm
from matplotlib import pyplot as plt

train = ireert.load_vm_flatbuffer_file(".vmfb_cache/fecb6c3f43b02e35e2b0cdb9af9407c9.vmfb", driver="cuda").fn2353
m = ireert.load_vm_flatbuffer_file(".vmfb_cache/fecb6c3f43b02e35e2b0cdb9af9407c9.vmfb", driver="cuda")
# reconstruct = ireert.load_vm_flatbuffer_file("reconstruct.vmfb", driver="cuda").fn6895


def init(shape):
    return (np.zeros(shape=shape).astype(np.float32),
     np.ones(shape=shape).astype(np.float32) * 0.001)

def init_dense(input, output):
    return lambda: [*init((1, output)), *init((input, output))]

encoder_1 = init_dense(784, 512)
encoder_2 = init_dense(512, 16)
encoder_3 = init_dense(512, 16)

decoder_1 = init_dense(16, 512)
decoder_2 = init_dense(512, 784)

param_initialisers = [decoder_2, decoder_1, encoder_3, encoder_2, encoder_1]

params = []

for i in range(3):
    for init_fn in param_initialisers:
        params.extend(init_fn())
    if i == 1:
        params.append(np.array(1.0).astype(np.float32))

num_epochs = 50000

bar = tqdm(zip(range(num_epochs), mnist()), total=num_epochs)

seed = np.array(0, dtype=np.uint64)

for i, (x, _) in bar:
    x = np.expand_dims(x, 1)
    print(x[0])
    [loss, *params, seed] = [x for x in train(*params, x, seed)]
    print(loss.to_host())
    print(params[0].to_host())
    break
    bar.set_description(f"Loss: {loss.to_host()}")
    # if (i) % 2500 == 0:
    #     seed, y_pred = reconstruct(seed, x, *params[:20])
    #     y_pred = y_pred.to_host()[0].reshape(28, 28)
    #     plt.imshow(y_pred, cmap="gray")
    #     plt.show()
